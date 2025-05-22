import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import os
import math

# --- Dummy classes (defined globally for fallback) ---
class GODag_Dummy:
    def __init__(self, obo_file=None):
        self.version = "dummy"
        self.data_version = "dummy"
        self.terms = {} 
        if obo_file and not os.path.exists(obo_file):
            print(f"Dummy GODag: Note - OBO file {obo_file} specified but not found.")

    def __getitem__(self, term_id):
        class DummyGOTerm:
            def __init__(self, term_id_inner):
                self.id = term_id_inner
                self.name = "dummy_term"
                self.namespace = "dummy_namespace"
                self.parents = set()
                self.children = set()
                self.level = 0
                self.depth = 0
                self.is_obsolete = False
                self.all_parents = set() 

            def get_all_parents(self): # Method used by propagate_go_terms
                return self.all_parents

        if term_id not in self.terms:
            self.terms[term_id] = DummyGOTerm(term_id)
        return self.terms[term_id]

    def get(self, term_id, default=None):
        return self.terms.get(term_id, default)
    
    def __len__(self):
        return len(self.terms)

    # Add a dummy get_ancestors if propagate_go_terms might call it directly on the DAG object
    def get_ancestors(self, term_id, include_self=True):
        ancestors = set()
        if include_self and self.get(term_id):
            ancestors.add(term_id)
        # For a dummy, we don't have real parent info to traverse
        return ancestors

class TermCounts_Dummy:
    def __init__(self, go2obj, annots):
        self.version = "dummy"
        self.go2obj = go2obj # Should be an instance of GODag_Dummy
        self.annots = annots 
        self.gocnts = self._count_terms()

    def _count_terms(self):
        cnts = Counter()
        if not isinstance(self.go2obj, GODag_Dummy): # Basic check
             print("Warning: TermCounts_Dummy received a non-dummy GODag object. This is unexpected.")
        for terms_set in self.annots.values():
            for term_id in terms_set:
                term_obj = self.go2obj.get(term_id)
                if term_obj and not term_obj.is_obsolete:
                    cnts[term_id] += 1
        return cnts
    
    def get_term_count(self, go_id):
        return self.gocnts.get(go_id, 0)
    
    # This dummy method was causing the confusion with the real API
    # The real TermCounts object uses .gocnts.keys()
    # def get_ids(self): 
    #     return list(self.gocnts.keys())

# --- Attempt to import real goatools ---
goatools_imported_successfully = False
GODag_To_Use = GODag_Dummy # Default to dummy
TermCounts_To_Use = TermCounts_Dummy # Default to dummy

try:
    from goatools.obo_parser import GODag as Real_GODag_Class
    from goatools.semantic import TermCounts as Real_TermCounts_Class
    goatools_imported_successfully = True
    GODag_To_Use = Real_GODag_Class
    TermCounts_To_Use = Real_TermCounts_Class
    print("Successfully imported real goatools library.")
except ImportError:
    print("Error: The 'goatools' library is not installed or not found in the Python path.")
    print("Please ensure it's installed in the correct environment (e.g., 'pip install goatools').")
    print("Hierarchical GO features will use dummy fallbacks and may not work as expected.")
except Exception as e:
    print(f"An unexpected error occurred during goatools import: {e}")
    print("Hierarchical GO features will use dummy fallbacks.")


# --- Configuration ---
INPUT_CSV_PATH = './data/PPI/raw/protein_list_with_features_clean.csv' # Updated to match user's log
OUTPUT_NPY_PATH = 'engineered_features_hierarchical.npy'
OUTPUT_CSV_PATH = 'engineered_features_hierarchical_with_ids.csv'
GO_OBO_PATH = 'go-basic.obo'

USE_PARENT_PROPAGATION = True
USE_IC_FEATURES = True
GO_SVD_COMPONENTS = 100
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# --- Helper Functions ---
def parse_go_terms_string(go_term_string):
    if pd.isna(go_term_string) or go_term_string == '':
        return set()
    return set(go_term_string.split(';'))

def propagate_go_terms(protein_go_terms_set, go_dag_instance):
    if not go_dag_instance or (hasattr(go_dag_instance, 'version') and go_dag_instance.version == "dummy" and not protein_go_terms_set):
        return protein_go_terms_set
    if not go_dag_instance or not (hasattr(go_dag_instance, 'get') or hasattr(go_dag_instance, '__getitem__')):
        if protein_go_terms_set : 
             print("Warning: GODag object for propagation is invalid or dummy. Skipping propagation.")
        return protein_go_terms_set

    propagated_terms = set()
    for term_id in protein_go_terms_set:
        term_obj = go_dag_instance.get(term_id) 
        if term_obj and not term_obj.is_obsolete:
            current_term_ancestors = set()
            if hasattr(term_obj, 'get_all_parents'): 
                current_term_ancestors = term_obj.get_all_parents()
                current_term_ancestors.add(term_id) 
            elif hasattr(go_dag_instance, 'get_ancestors'): 
                 current_term_ancestors = go_dag_instance.get_ancestors(term_id, include_self=True)
            else: 
                current_term_ancestors.add(term_id)
            propagated_terms.update(current_term_ancestors)
        else:
            propagated_terms.add(term_id)
    return propagated_terms

def calculate_aac(sequence):
    if pd.isna(sequence) or sequence == '':
        return {aa: 0.0 for aa in AMINO_ACIDS}
    valid_sequence = ''.join(filter(lambda char: char in AMINO_ACIDS, sequence.upper()))
    counts = Counter(valid_sequence)
    total_aas = len(valid_sequence)
    aac_vector = {}
    if total_aas > 0:
        for aa in AMINO_ACIDS:
            aac_vector[aa] = counts.get(aa, 0) / total_aas
    else:
        for aa in AMINO_ACIDS:
            aac_vector[aa] = 0.0
    return aac_vector

def main():
    print(f"Loading data from {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV_PATH}. Creating dummy data.")

    go_dag_instance = None
    obo_file_exists = os.path.exists(GO_OBO_PATH)

    if USE_PARENT_PROPAGATION or USE_IC_FEATURES:
        if not goatools_imported_successfully:
            print("Using dummy GODag because goatools library import failed.")
            go_dag_instance = GODag_Dummy(GO_OBO_PATH)
        elif not obo_file_exists:
            print(f"Error: GO OBO file not found at {GO_OBO_PATH}. Using dummy GODag.")
            go_dag_instance = GODag_Dummy(GO_OBO_PATH)
        else:
            print(f"Loading Gene Ontology from {GO_OBO_PATH} using real goatools...")
            try:
                go_dag_instance = GODag_To_Use(GO_OBO_PATH, optional_attrs=['relationship'])
                if not hasattr(go_dag_instance, 'version') or go_dag_instance.version == "dummy":
                     print("Critical Warning: Loaded GODag is a dummy instance despite expecting real goatools. Check import logic.")
                else:
                     print(f"  GO DAG loaded. Version: {getattr(go_dag_instance, 'version', 'N/A')}, Terms: {len(go_dag_instance) if hasattr(go_dag_instance, '__len__') else 'N/A'}.")
            except Exception as e:
                print(f"Error loading real GODag with goatools: {e}. Using dummy GODag as fallback.")
                go_dag_instance = GODag_Dummy(GO_OBO_PATH)
    
    print("Engineering GO term features...")
    df['parsed_GO_terms'] = df['GO_term'].apply(parse_go_terms_string)
    all_protein_go_sets_for_ic = {} 

    if USE_PARENT_PROPAGATION:
        if go_dag_instance and (not hasattr(go_dag_instance, 'version') or go_dag_instance.version != "dummy"):
            print("  Applying parent term propagation using real GODag...")
        else:
            print("  Applying parent term propagation using dummy GODag (results may be basic).")
        
        propagated_go_column = []
        for index, row in df.iterrows():
            propagated_set = propagate_go_terms(row['parsed_GO_terms'], go_dag_instance)
            propagated_go_column.append(propagated_set)
            all_protein_go_sets_for_ic[row['protein_id']] = propagated_set
        df['processed_GO_terms'] = propagated_go_column
    else:
        print("  Skipping parent term propagation.")
        df['processed_GO_terms'] = df['parsed_GO_terms']
        for index, row in df.iterrows(): 
            all_protein_go_sets_for_ic[row['protein_id']] = row['parsed_GO_terms']

    mlb = MultiLabelBinarizer(sparse_output=True)
    go_multi_hot_sparse = mlb.fit_transform(df['processed_GO_terms'])
    go_term_names_full = [f'GO_{cls.replace(":", "_")}' for cls in mlb.classes_]
    print(f"  Number of unique GO terms for Multi-Hot: {len(mlb.classes_)}")
    
    if len(mlb.classes_) > GO_SVD_COMPONENTS and GO_SVD_COMPONENTS > 0:
        print(f"  Applying Truncated SVD (components: {GO_SVD_COMPONENTS})...")
        svd = TruncatedSVD(n_components=GO_SVD_COMPONENTS, random_state=42)
        go_features_reduced = svd.fit_transform(go_multi_hot_sparse)
        go_feature_names_svd = [f'GO_SVD_{i}' for i in range(GO_SVD_COMPONENTS)]
        go_features_df = pd.DataFrame(go_features_reduced, columns=go_feature_names_svd, index=df.index)
    elif len(mlb.classes_) > 0:
        print("  Using full Multi-Hot Encoded GO features (SVD not applied).")
        go_features_df = pd.DataFrame(go_multi_hot_sparse.toarray(), columns=go_term_names_full, index=df.index)
    else:
        go_features_df = pd.DataFrame(np.zeros((len(df), 0)), index=df.index)

    ic_features_df = pd.DataFrame(index=df.index)
    if USE_IC_FEATURES:
        if goatools_imported_successfully and go_dag_instance and \
           (hasattr(go_dag_instance, 'version') and go_dag_instance.version != "dummy"):
            print("  Calculating Information Content (IC) features using real goatools...")
            try:
                term_counts_instance = TermCounts_To_Use(go_dag_instance, all_protein_go_sets_for_ic)
                go_ic_map = {}
                total_proteins_with_any_go = sum(1 for go_set in all_protein_go_sets_for_ic.values() if go_set)

                if total_proteins_with_any_go > 0:
                    # Corrected line: iterate over term_counts_instance.gocnts.keys()
                    for term_id in term_counts_instance.gocnts.keys(): 
                        proteins_with_term_t_count = sum(1 for protein_annot_set in all_protein_go_sets_for_ic.values() if term_id in protein_annot_set)
                        if proteins_with_term_t_count > 0:
                            prob = proteins_with_term_t_count / total_proteins_with_any_go
                            go_ic_map[term_id] = -math.log2(prob)
                        else: 
                            go_ic_map[term_id] = 0 
                else:
                    print("    No proteins with GO terms found for IC calculation denominator.")
                
                ic_sum_list, ic_mean_list = [], []
                for protein_id in df['protein_id']:
                    protein_go_terms = all_protein_go_sets_for_ic.get(protein_id, set())
                    current_protein_ics = [go_ic_map.get(term, 0) for term in protein_go_terms if go_ic_map.get(term, 0) > 0]
                    ic_sum_list.append(sum(current_protein_ics) if current_protein_ics else 0.0)
                    ic_mean_list.append(np.mean(current_protein_ics) if current_protein_ics else 0.0)
                
                ic_features_df['GO_IC_Sum'] = ic_sum_list
                ic_features_df['GO_IC_Mean'] = ic_mean_list
                print(f"    Generated IC features: {ic_features_df.columns.tolist()}")
            except AttributeError as e_attr: # Catch AttributeError specifically
                print(f"    AttributeError calculating IC features: {e_attr}. This might indicate an issue with TermCounts API.")
                print(f"    TermCounts object type: {type(term_counts_instance)}")
                print(f"    Attributes: {dir(term_counts_instance)}")

            except Exception as e:
                print(f"    Error calculating IC features with real goatools: {e}. Skipping.")
                ic_features_df = pd.DataFrame(np.zeros((len(df), 0)), index=df.index)
        else:
            print("  Skipping IC features (real goatools not available or using dummy GODag).")
            ic_features_df = pd.DataFrame(np.zeros((len(df), 0)), index=df.index)

    print("Engineering sequence features...")
    df['sequence_length'] = df['sequence'].apply(lambda x: len(x) if pd.notna(x) else 0)
    # Apply log1p transformation to sequence_length
    print("  Applying log1p transformation to sequence_length...")
    df['sequence_length'] = np.log1p(df['sequence_length']) 
    seq_len_features = df[['sequence_length']].copy() # This will now contain the log-transformed values

    aac_data = df['sequence'].apply(calculate_aac)
    aac_df = pd.DataFrame.from_records(aac_data, index=df.index)
    aac_feature_names = [f'AAC_{aa}' for aa in AMINO_ACIDS]
    aac_df.columns = aac_feature_names
    
    print("Incorporating existing numerical and binary features...")
    graph_features_df = df[['degree', 'betweenness', 'closeness']].copy()
    
    # Apply log1p transformation to graph features
    print("  Applying log1p transformation to degree, betweenness, closeness...")
    for col in ['degree', 'betweenness', 'closeness']:
        # Ensure column exists and is numeric before transformation
        if col in graph_features_df.columns:
            # Check for non-numeric types that might cause issues with log1p
            if pd.api.types.is_numeric_dtype(graph_features_df[col]):
                graph_features_df[col] = np.log1p(graph_features_df[col].astype(float)) # Ensure float for log1p
            else:
                print(f"    Warning: Column '{col}' is not numeric. Skipping log1p transformation for this column.")
        else:
            print(f"    Warning: Column '{col}' not found in graph_features_df. Skipping log1p transformation.")

    binary_features_df = df[['has_GO_term', 'has_sequence']].copy()

    print("Combining all features...")
    all_features_list = [go_features_df, ic_features_df, seq_len_features, aac_df, graph_features_df, binary_features_df]
    all_features_list_filtered = [f_df for f_df in all_features_list if not f_df.empty]

    if not all_features_list_filtered:
        print("Error: No features were generated. Exiting.")
        return
    all_features_df = pd.concat(all_features_list_filtered, axis=1).fillna(0)

    print("Scaling numerical features...")
    features_to_scale_names = []
    # GO SVD features (already handled by previous logic)
    if not go_features_df.empty and ('GO_SVD_0' in go_features_df.columns): 
        features_to_scale_names.extend([col for col in go_features_df.columns if 'GO_SVD' in col])
    # IC features (already handled by previous logic)
    if not ic_features_df.empty: 
        features_to_scale_names.extend(ic_features_df.columns.tolist())
    
    # Add the (now log-transformed) sequence_length and graph metrics
    features_to_scale_names.extend(['sequence_length']) 
    features_to_scale_names.extend(aac_df.columns.tolist()) # AAC features
    features_to_scale_names.extend(['degree', 'betweenness', 'closeness']) # Graph features
    
    valid_features_to_scale = []
    for name in features_to_scale_names:
        if name in all_features_df.columns and all_features_df[name].nunique() > 1:
            valid_features_to_scale.append(name)
        elif name in all_features_df.columns:
            # This specific print might be noisy if many AAC features have single values in small test sets
            # For production, you might only print if it's NOT an AAC feature or if verbosity is high.
            if not name.startswith("AAC_") or all_features_df[name].nunique() <=1 : 
                 print(f"  Skipping scaling for column '{name}' due to zero or single unique value.")
    
    if valid_features_to_scale:
        scaler = StandardScaler()
        all_features_df[valid_features_to_scale] = scaler.fit_transform(all_features_df[valid_features_to_scale])
        print(f"  Scaled columns: {valid_features_to_scale}")
    else:
        print("  No features with sufficient variance found for scaling.")

    final_feature_matrix = all_features_df.values
    output_df_csv = pd.concat([df[['protein_name', 'protein_id']], all_features_df], axis=1)

    print(f"Saving engineered features matrix to {OUTPUT_NPY_PATH}...")
    np.save(OUTPUT_NPY_PATH, final_feature_matrix)
    print(f"Saving engineered features with IDs and headers to {OUTPUT_CSV_PATH}...")
    output_df_csv.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\nFeature engineering complete!")
    print(f"  Shape of final feature matrix: {final_feature_matrix.shape}")
    print(f"  Number of features: {len(all_features_df.columns)}")

if __name__ == '__main__':
    main()
