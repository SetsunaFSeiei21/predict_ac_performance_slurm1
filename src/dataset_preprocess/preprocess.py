import argparse, os
from convertos import convert_json2csv
from message_generate import parse_netlist
from netlist_generate import generate_pin_level_netlist, generate_device_level_netlist

def preprocess_data(args: argparse.Namespace) -> None:
    
    root_path = args.Data_Path
    circuit_name = args.Circuit_Name
    circuit_dir = os.path.join(os.path.abspath(root_path), circuit_name)
    mission_type = args.Mission_Type
    mission_dir = os.path.join(circuit_dir, mission_type)
    netlist_file_path = os.path.join(circuit_dir, f"{circuit_name}_netlist.txt")
    
    assert os.path.exists(os.path.abspath(root_path)), f"Data path:{root_path} does not exist!"
    assert os.path.exists(circuit_dir), f"Circuit {circuit_name} does not exist!"
    assert os.path.exists(mission_dir), f"{mission_type} data does not exist!"
    assert os.path.exists(netlist_file_path), f"Netlist of circuit:{circuit_name} does not exist!"
    
    convert_json2csv(mission_dir)
    device_messages = parse_netlist(netlist_file_path)
    generate_device_level_netlist(device_messages, circuit_dir)
    generate_pin_level_netlist(device_messages, circuit_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Data_Path", required=True, type=str, help="The root path of the data.")
    parser.add_argument("--Circuit_Name", required=True, type=str, help="The circuit name.")
    parser.add_argument("--Mission_Type", required=True, type=str, help="Mission type, source or target.")
    args = parser.parse_args()
    
    preprocess_data(args)