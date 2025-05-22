import argparse
from preprocess_ppi import PPIPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Preprocess PPI network data')
    
    # Add command-line arguments
    parser.add_argument('--ppi_file', type=str, default='4932.protein.links.v12.0.txt',
                        help='File containing protein-protein interactions')
    parser.add_argument('--info_file', type=str, default='4932.protein.info.v12.0.txt',
                        help='File containing protein information')
    parser.add_argument('--alias_file', type=str, default='4932.protein.aliases.v12.0.txt',
                        help='File containing protein aliases/ID mappings')
    parser.add_argument('--fasta_file', type=str, default='UP000002311_559292.fasta',
                        help='FASTA file containing protein sequences')
    parser.add_argument('--gaf_file', type=str, default='sgd.gaf',
                        help='GAF file containing GO annotations')
    parser.add_argument('--confidence', type=int, default=700,
                        help='Minimum confidence score for interactions (0-1000)')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save processed data')
    
    args = parser.parse_args()
    
    # Create preprocessor with provided arguments
    preprocessor = PPIPreprocessor(
        ppi_file=args.ppi_file,
        info_file=args.info_file,
        alias_file=args.alias_file,
        fasta_file=args.fasta_file,
        gaf_file=args.gaf_file,
        confidence_threshold=args.confidence,
        output_dir=args.output_dir
    )
    
    # Run preprocessing pipeline
    print(f"Starting preprocessing with confidence threshold {args.confidence}")
    results = preprocessor.run_pipeline()
    
    # Print summary
    print("\nPreprocessing completed successfully!")
    print("\nNetwork Statistics:")
    for key, value in results['stats'].items():
        print(f"{key}: {value}")
    
    print(f"\nProcessed files saved to {args.output_dir}/")

if __name__ == "__main__":
    main()