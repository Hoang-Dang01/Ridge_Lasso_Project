# src/data_generator.py
import random
import os

def generate_logistics_data(n=300, filename='logistics_data.csv'):
    header = ["Input_Qty", "Output_Qty", "Lead_Time", "Warehouse_Temp", "Promo_Flag", "Inventory_Level"]
    
    # Save the file to the '../data/' folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(",".join(header) + "\n")
        for _ in range(n):
            in_qty = random.randint(100, 1000)
            out_qty = random.randint(50, 900)
            lead_time = random.randint(1, 15)
            temp = random.uniform(20, 40) # Noise variable
            promo = random.randint(0, 1)
            
            # Inventory_Level = 0.7*In_Qty - 0.8*Out_Qty + 5.0*Lead_Time + 20.0*Promo + Noise
            level = (in_qty * 0.7) - (out_qty * 0.8) + (lead_time * 5.0) + (promo * 20.0) + random.normalvariate(0, 5)
            row = [in_qty, out_qty, lead_time, temp, promo, level]
            f.write(",".join(map(str, row)) + "\n")
    print("Successfully generated dataset!")

if __name__ == "__main__":
    generate_logistics_data(n=1000)
