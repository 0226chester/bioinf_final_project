import os
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("coverage_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("coverage_analysis")

class CoverageAnalyzer:
    def __init__(self, data_dir="processed_data"):
        """
        Initialize the coverage analyzer.
        
        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = data_dir
        
        # Initialize data structures
        self.graph = None
        self.protein_sequences = None
        self.protein_to_go = None
        self.string_to_alias = None
        self.alias_to_string = None
        
        # Coverage statistics
        self.stats = {
            'total_proteins': 0,
            'go_coverage': 0,
            'sequence_coverage': 0,
            'both_coverage': 0,
            'only_go': 0,
            'only_sequence': 0,
            'missing_both': 0
        }
    
    def load_data(self):
        """Load all required data files."""
        logger.info(f"Loading data from {self.data_dir}")
        
        # Load network graph
        graph_path = os.path.join(self.data_dir, 'ppi_network.gpickle')
        logger.info(f"Loading network from {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Loaded network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Load protein sequences
        seq_path = os.path.join(self.data_dir, 'protein_sequences.pkl')
        logger.info(f"Loading protein sequences from {seq_path}")
        with open(seq_path, 'rb') as f:
            self.protein_sequences = pickle.load(f)
        logger.info(f"Loaded {len(self.protein_sequences)} protein sequences")
        
        # Load GO annotations
        go_path = os.path.join(self.data_dir, 'protein_to_go.pkl')
        logger.info(f"Loading GO annotations from {go_path}")
        with open(go_path, 'rb') as f:
            self.protein_to_go = pickle.load(f)
        logger.info(f"Loaded GO annotations for {len(self.protein_to_go)} proteins")
        
        # Load alias mappings
        alias_path = os.path.join(self.data_dir, 'string_to_alias.pkl')
        logger.info(f"Loading protein aliases from {alias_path}")
        with open(alias_path, 'rb') as f:
            self.string_to_alias = pickle.load(f)
        
        # Load reverse alias mappings if available
        reverse_alias_path = os.path.join(self.data_dir, 'alias_to_string.pkl')
        if os.path.exists(reverse_alias_path):
            logger.info(f"Loading reverse protein aliases from {reverse_alias_path}")
            with open(reverse_alias_path, 'rb') as f:
                self.alias_to_string = pickle.load(f)
        else:
            logger.warning(f"Reverse alias mapping not found at {reverse_alias_path}")
            self.alias_to_string = {}
    
    def analyze_coverage(self):
        """Analyze coverage of GO terms and sequences for proteins in the graph."""
        logger.info("Analyzing protein coverage")
        
        # Get all proteins in the graph
        proteins = list(self.graph.nodes())
        self.stats['total_proteins'] = len(proteins)
        logger.info(f"Analyzing coverage for {self.stats['total_proteins']} proteins")
        
        # For detailed analysis
        proteins_with_go = set()
        proteins_with_sequence = set()
        
        # Mapping counters
        go_mapping_sources = defaultdict(int)
        sequence_mapping_sources = defaultdict(int)
        
        # Check each protein
        for protein in proteins:
            has_go = False
            has_sequence = False
            
            # Check for direct GO mapping
            if protein in self.protein_to_go:
                has_go = True
                go_mapping_sources['direct'] += 1
            
            # Check for GO mapping via aliases
            if not has_go and protein in self.string_to_alias:
                for source, aliases in self.string_to_alias[protein].items():
                    if source in ['UniProt_DR_SGD', 'SGD_ID']:
                        for alias in aliases:
                            if alias in self.protein_to_go:
                                has_go = True
                                go_mapping_sources[source] += 1
                                break
                    if has_go:
                        break
            
            # Check for direct sequence mapping
            if protein in self.protein_sequences:
                has_sequence = True
                sequence_mapping_sources['direct'] += 1
            
            # Check for sequence mapping via aliases
            if not has_sequence and protein in self.string_to_alias:
                for source, aliases in self.string_to_alias[protein].items():
                    if source in ['Ensembl_UniProt', 'UniProt_AC']:
                        for alias in aliases:
                            if alias in self.protein_sequences:
                                has_sequence = True
                                sequence_mapping_sources[source] += 1
                                break
                    if has_sequence:
                        break
            
            # Update tracking sets
            if has_go:
                proteins_with_go.add(protein)
            
            if has_sequence:
                proteins_with_sequence.add(protein)
        
        # Calculate coverage statistics
        proteins_with_both = proteins_with_go.intersection(proteins_with_sequence)
        proteins_only_go = proteins_with_go - proteins_with_sequence
        proteins_only_sequence = proteins_with_sequence - proteins_with_go
        proteins_missing_both = set(proteins) - proteins_with_go - proteins_with_sequence
        
        self.stats['go_coverage'] = len(proteins_with_go)
        self.stats['sequence_coverage'] = len(proteins_with_sequence)
        self.stats['both_coverage'] = len(proteins_with_both)
        self.stats['only_go'] = len(proteins_only_go)
        self.stats['only_sequence'] = len(proteins_only_sequence)
        self.stats['missing_both'] = len(proteins_missing_both)
        
        # Log statistics
        logger.info("Coverage Statistics:")
        logger.info(f"Total proteins in network: {self.stats['total_proteins']}")
        logger.info(f"Proteins with GO terms: {self.stats['go_coverage']} ({self.stats['go_coverage']/self.stats['total_proteins']*100:.2f}%)")
        logger.info(f"Proteins with sequences: {self.stats['sequence_coverage']} ({self.stats['sequence_coverage']/self.stats['total_proteins']*100:.2f}%)")
        logger.info(f"Proteins with both: {self.stats['both_coverage']} ({self.stats['both_coverage']/self.stats['total_proteins']*100:.2f}%)")
        logger.info(f"Proteins with only GO: {self.stats['only_go']} ({self.stats['only_go']/self.stats['total_proteins']*100:.2f}%)")
        logger.info(f"Proteins with only sequence: {self.stats['only_sequence']} ({self.stats['only_sequence']/self.stats['total_proteins']*100:.2f}%)")
        logger.info(f"Proteins missing both: {self.stats['missing_both']} ({self.stats['missing_both']/self.stats['total_proteins']*100:.2f}%)")
        
        # Log mapping sources
        logger.info("\nGO Mapping Sources:")
        for source, count in go_mapping_sources.items():
            logger.info(f"  {source}: {count} ({count/self.stats['go_coverage']*100:.2f}%)")
        
        logger.info("\nSequence Mapping Sources:")
        for source, count in sequence_mapping_sources.items():
            logger.info(f"  {source}: {count} ({count/self.stats['sequence_coverage']*100:.2f}%)")
        
        return self.stats
    
    def analyze_go_distribution(self):
        """Analyze the distribution of GO terms across proteins."""
        if not self.protein_to_go:
            logger.error("GO data not loaded!")
            return None
        
        logger.info("Analyzing GO term distribution")
        
        # GO term counts by aspect
        bp_counts = []
        mf_counts = []
        cc_counts = []
        
        for protein, go_dict in self.protein_to_go.items():
            bp_counts.append(len(go_dict.get('P', set())))
            mf_counts.append(len(go_dict.get('F', set())))
            cc_counts.append(len(go_dict.get('C', set())))
        
        # Calculate statistics
        stats = {
            'BP': {
                'mean': sum(bp_counts) / len(bp_counts) if bp_counts else 0,
                'max': max(bp_counts) if bp_counts else 0,
                'min': min(bp_counts) if bp_counts else 0,
                'total': sum(bp_counts) if bp_counts else 0
            },
            'MF': {
                'mean': sum(mf_counts) / len(mf_counts) if mf_counts else 0,
                'max': max(mf_counts) if mf_counts else 0,
                'min': min(mf_counts) if mf_counts else 0,
                'total': sum(mf_counts) if mf_counts else 0
            },
            'CC': {
                'mean': sum(cc_counts) / len(cc_counts) if cc_counts else 0,
                'max': max(cc_counts) if cc_counts else 0,
                'min': min(cc_counts) if cc_counts else 0,
                'total': sum(cc_counts) if cc_counts else 0
            }
        }
        
        # Log statistics
        logger.info("GO Term Distribution:")
        logger.info(f"Biological Process: avg={stats['BP']['mean']:.2f}, min={stats['BP']['min']}, max={stats['BP']['max']}, total={stats['BP']['total']}")
        logger.info(f"Molecular Function: avg={stats['MF']['mean']:.2f}, min={stats['MF']['min']}, max={stats['MF']['max']}, total={stats['MF']['total']}")
        logger.info(f"Cellular Component: avg={stats['CC']['mean']:.2f}, min={stats['CC']['min']}, max={stats['CC']['max']}, total={stats['CC']['total']}")
        
        return stats
    
    def visualize_coverage(self, output_file='coverage_analysis.png'):
        """Visualize coverage statistics."""
        logger.info(f"Generating coverage visualization to {output_file}")
        
        # Create bar chart of coverage
        labels = ['GO Terms', 'Sequences', 'Both', 'Only GO', 'Only Seq', 'Missing Both']
        values = [
            self.stats['go_coverage'] / self.stats['total_proteins'] * 100,
            self.stats['sequence_coverage'] / self.stats['total_proteins'] * 100,
            self.stats['both_coverage'] / self.stats['total_proteins'] * 100,
            self.stats['only_go'] / self.stats['total_proteins'] * 100,
            self.stats['only_sequence'] / self.stats['total_proteins'] * 100,
            self.stats['missing_both'] / self.stats['total_proteins'] * 100
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e74c3c', '#95a5a6'])
        
        # Add labels and title
        plt.xlabel('Coverage Category')
        plt.ylabel('Percentage (%)')
        plt.title('Protein Coverage in PPI Network')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Set y-axis limit
        plt.ylim(0, 105)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, output_file))
        logger.info(f"Saved coverage visualization to {os.path.join(self.data_dir, output_file)}")
    
    def generate_coverage_report(self):
        """Generate a comprehensive coverage report."""
        report_path = os.path.join(self.data_dir, 'coverage_report.txt')
        logger.info(f"Generating coverage report to {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("=== PROTEIN COVERAGE ANALYSIS REPORT ===\n\n")
            
            f.write("COVERAGE STATISTICS:\n")
            f.write(f"Total proteins in network: {self.stats['total_proteins']}\n")
            f.write(f"Proteins with GO terms: {self.stats['go_coverage']} ({self.stats['go_coverage']/self.stats['total_proteins']*100:.2f}%)\n")
            f.write(f"Proteins with sequences: {self.stats['sequence_coverage']} ({self.stats['sequence_coverage']/self.stats['total_proteins']*100:.2f}%)\n")
            f.write(f"Proteins with both: {self.stats['both_coverage']} ({self.stats['both_coverage']/self.stats['total_proteins']*100:.2f}%)\n")
            f.write(f"Proteins with only GO: {self.stats['only_go']} ({self.stats['only_go']/self.stats['total_proteins']*100:.2f}%)\n")
            f.write(f"Proteins with only sequence: {self.stats['only_sequence']} ({self.stats['only_sequence']/self.stats['total_proteins']*100:.2f}%)\n")
            f.write(f"Proteins missing both: {self.stats['missing_both']} ({self.stats['missing_both']/self.stats['total_proteins']*100:.2f}%)\n\n")
            
            f.write("IMPLICATIONS FOR LINK PREDICTION:\n")
            f.write("- For sequence-based features, you can use data for ")
            f.write(f"{self.stats['sequence_coverage']/self.stats['total_proteins']*100:.2f}% of the network proteins.\n")
            f.write("- For GO-based features, you can use data for ")
            f.write(f"{self.stats['go_coverage']/self.stats['total_proteins']*100:.2f}% of the network proteins.\n")
            f.write("- For hybrid approaches using both GO and sequence, you can use data for ")
            f.write(f"{self.stats['both_coverage']/self.stats['total_proteins']*100:.2f}% of the network proteins.\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            # Add recommendations based on coverage results
            if self.stats['both_coverage']/self.stats['total_proteins'] > 0.8:
                f.write("- Coverage for both GO and sequence data is high (>80%). A hybrid approach is recommended.\n")
            elif self.stats['go_coverage']/self.stats['total_proteins'] > 0.8:
                f.write("- GO term coverage is high (>80%). Consider GO-based features as primary approach.\n")
            elif self.stats['sequence_coverage']/self.stats['total_proteins'] > 0.8:
                f.write("- Sequence coverage is high (>80%). Consider sequence-based features as primary approach.\n")
            else:
                f.write("- Coverage for both data types is limited. Consider primarily using topological features.\n")
            
            if self.stats['missing_both']/self.stats['total_proteins'] > 0.1:
                f.write(f"- {self.stats['missing_both']} proteins ({self.stats['missing_both']/self.stats['total_proteins']*100:.2f}%) ")
                f.write("lack both GO and sequence data. Consider handling these proteins specially in your model.\n")
        
        logger.info(f"Coverage report saved to {report_path}")
    
    def run_analysis(self):
        """Run the complete coverage analysis pipeline."""
        logger.info("Starting coverage analysis")
        
        try:
            # Load data
            self.load_data()
            
            # Analyze coverage
            self.analyze_coverage()
            
            # Analyze GO distribution
            self.analyze_go_distribution()
            
            # Generate visualizations
            self.visualize_coverage()
            
            # Generate report
            self.generate_coverage_report()
            
            logger.info("Coverage analysis completed successfully")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Create analyzer with default settings
    analyzer = CoverageAnalyzer(data_dir="processed_data")
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Print summary
    print("\nCoverage Analysis Summary:")
    print(f"Total proteins in network: {results['total_proteins']}")
    print(f"Proteins with GO terms: {results['go_coverage']} ({results['go_coverage']/results['total_proteins']*100:.2f}%)")
    print(f"Proteins with sequences: {results['sequence_coverage']} ({results['sequence_coverage']/results['total_proteins']*100:.2f}%)")
    print(f"Proteins with both: {results['both_coverage']} ({results['both_coverage']/results['total_proteins']*100:.2f}%)")
    print(f"Proteins with only GO: {results['only_go']} ({results['only_go']/results['total_proteins']*100:.2f}%)")
    print(f"Proteins with only sequence: {results['only_sequence']} ({results['only_sequence']/results['total_proteins']*100:.2f}%)")
    print(f"Proteins missing both: {results['missing_both']} ({results['missing_both']/results['total_proteins']*100:.2f}%)")
    print("\nDetailed report saved to processed_data/coverage_report.txt")