import os
import sys
import numpy as np
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Check the shapes of logits files in a directory")
    parser.add_argument('--logits_dir', type=str, default='/mnt/storage/data/logits/synth/logits',
                        help='Directory containing logits files')
    parser.add_argument('--pattern', type=str, default='_logits.npy',
                        help='File pattern to match')
    parser.add_argument('--save_report', action='store_true',
                        help='Save the report to a file')
    parser.add_argument('--report_path', type=str, default='/mnt/storage/data/logits_report.txt',
                        help='Path to save the report')
    args = parser.parse_args()
    
    print(f"Analyzing logits files in: {args.logits_dir}")
    
    # Dictionary to store shape frequencies
    shape_counts = defaultdict(int)
    shape_examples = defaultdict(list)
    total_files = 0
    problematic_files = []
    
    # Walk through all subdirectories in the logits directory
    for root, dirs, files in os.walk(args.logits_dir):
        for file in files:
            if args.pattern in file:
                total_files += 1
                file_path = os.path.join(root, file)
                
                try:
                    # Load the logits file
                    logits = np.load(file_path, allow_pickle=False)
                    
                    # Get the shape as a string
                    shape_str = str(logits.shape)
                    
                    # Record the shape
                    shape_counts[shape_str] += 1
                    
                    # Store up to 5 examples for each shape
                    if len(shape_examples[shape_str]) < 5:
                        shape_examples[shape_str].append(file_path)
                    
                    # Check if the shape is problematic
                    if len(logits.shape) != 3 or logits.shape[0] != 1:
                        problematic_files.append((file_path, shape_str))
                        
                except Exception as e:
                    problematic_files.append((file_path, f"Error: {str(e)}"))
    
    # Generate report
    report = []
    report.append(f"Logits Shape Analysis Report")
    report.append(f"----------------------------")
    report.append(f"Total logits files analyzed: {total_files}")
    report.append(f"Number of distinct shapes: {len(shape_counts)}")
    report.append("")
    
    # Sort shapes by frequency (most common first)
    sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
    
    report.append("Shape Distribution:")
    report.append("------------------")
    for shape, count in sorted_shapes:
        report.append(f"{shape}: {count} files ({count/total_files*100:.2f}%)")
    report.append("")
    
    # Example files for each shape
    report.append("Examples for each shape:")
    report.append("-----------------------")
    for shape, examples in shape_examples.items():
        report.append(f"{shape}:")
        for example in examples:
            report.append(f"  - {os.path.basename(example)}")
        report.append("")
    
    # Problematic files
    if problematic_files:
        report.append("Problematic Files:")
        report.append("-----------------")
        for file_path, issue in problematic_files:
            report.append(f"{os.path.basename(file_path)}: {issue}")
        report.append("")
    
    # Print the report
    for line in report:
        print(line)
    
    # Save the report if requested
    if args.save_report:
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, 'w') as f:
            for line in report:
                f.write(line + "\n")
        print(f"Report saved to: {args.report_path}")

if __name__ == "__main__":
    main()
