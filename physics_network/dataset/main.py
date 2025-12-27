# main.py
from config_param import ConfigParam as Config
from dataset.process_maps import run as run_maps
from dataset.process_met import run as run_met
from dataset.process_labels import run as run_labels

def main():
    # 폴더 생성
    Config.make_dirs()
    print("=== DeepONet Preprocessing Started ===\n")
    
    # 1. Maps 생성
    run_maps()
    
    # 2. Met 생성 (DF 반환)
    met_df = run_met()
    
    # 3. Label 생성
    run_labels(met_df)
    
    print("\n=== All Done! Data ready in './processed_data' ===")

if __name__ == "__main__":
    main()