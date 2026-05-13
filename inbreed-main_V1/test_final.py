import pandas as pd
import numpy as np
from app import calculate, standardize_input

def test():
    raw_df = pd.DataFrame({
        "Animal_ID": ["I", "P", "C", "D", "X", "B", "A", "E", "F"],
        "Sire_ID": ["-", "-", "I", "I", "I", "D", "B", "B", "D"],
        "Dam_ID": ["-", "-", "P", "P", "P", "C", "C", "-", "-"],
    })
    
    internal = standardize_input(raw_df, "Animal_ID", "Sire_ID", "Dam_ID")
    _, res_df, _ = calculate(internal)
    
    # Check key values
    results = dict(zip(res_df["Animal_ID"], res_df["Inbreeding_%"]))
    
    print(f"I: {results.get('I')}%")
    print(f"B: {results.get('B')}% (Expected 25.0)")
    print(f"A: {results.get('A')}% (Expected 37.5)")
    
    assert results.get("I") == 0.0
    assert results.get("P") == 0.0
    assert results.get("B") == 25.0
    assert results.get("A") == 37.5
    print("ALL TESTS PASSED\!")

if __name__ == "__main__":
    test()
