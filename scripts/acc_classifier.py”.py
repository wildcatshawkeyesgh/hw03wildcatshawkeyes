#!/usr/bin/env python3
"""
Script: acc_classifier.py”
Package: hw03wildcatshawkeyes
Course: CPE 487/587 - Machine Learning Tools
Homework: HW03

Usage:
    python scripts/acc_classifier.py”.py
"""

from hw03wildcatshawkeyes import example_function

def main():
    # Test data
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Run the function
    result = example_function(test_data)
    
    print(f"Input: {test_data}")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
