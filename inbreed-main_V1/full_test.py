import pandas as pd
import numpy as np
import app

def test():
    # Test sample data
    df = app.contoh_dari_gambar()
    # Mock clean_id for simple strings
    df['Animal_ID'] = df['Animal_ID'].astype(str)
    df['Sire_ID'] = df['Sire_ID'].replace('-', None)
    df['Dam_ID'] = df['Dam_ID'].replace('-', None)
    
    _, result, _ = app.calculate(df)
    
    # Check results
    res_dict = result.set_index('Animal_ID')['Inbreeding_%'].to_dict()
    print("Test Results:", res_dict)
    
    assert round(res_dict['B'], 2) == 25.0
    assert round(res_dict['A'], 2) == 37.5
    print("Test passed\!")

if __name__ == "__main__":
    test()
