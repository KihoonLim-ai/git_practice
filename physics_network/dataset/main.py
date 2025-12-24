# main.py
from config_param import ConfigParam as Config
import process_maps
import process_met
import process_labels

def main():
    # 폴더 생성
    Config.make_dirs()
    print("=== DeepONet Preprocessing Started ===\n")
    
    # 1. Maps 생성
    process_maps.run()
    
    # 2. Met 생성 (DF 반환)
    met_df = process_met.run()
    
    # 3. Label 생성
    process_labels.run(met_df)
    
    print("\n=== All Done! Data ready in './processed_data' ===")

if __name__ == "__main__":
    main()