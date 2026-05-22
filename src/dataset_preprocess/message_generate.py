import os, json
from typing import List, Dict, Any, Optional

def parse_netlist(netlist_file_path: str) -> List[Dict[str, Any]]:
    
    device_message_path = os.path.join(os.path.dirname(os.path.abspath(netlist_file_path)), 'device_messages.json')
    if os.path.exists(device_message_path):
        with open(device_message_path, mode='r', encoding='utf-8') as f:
            return json.load(f)
    content_lst: List[str] = []
    with open(netlist_file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith('.') and not line.startswith('*'):
                content_lst.append(line)
    devices = []
    for content in content_lst:
        if content.endswith('\n'):
            content = content.replace('\n', '')
        tokens = content.split()
        name = tokens[0]
        if name.startswith('M'):
            mos_type = name[1]
            device = {
                "name": name,
                "kind": "nmos" if mos_type.lower() == 'n' else "pmos",
                "pin_type": ['D', 'G', 'S', 'B'],
                "pins": tokens[1:5],
                "params": {}
            }
            for token in tokens[6:]:
                if '=' in token:
                    k, v = token.split('=', 1)
                    device['params'][k] = v
            devices.append(device)
        elif name.startswith("R"):
            device = {
                "name": name,
                "kind": "resistor",
                "pin_type": ['res_plus', 'res_minus'],
                "pins": tokens[1:3],
                "params": {
                    "res": tokens[3]
                }
            }
            devices.append(device)
        elif name.startswith("C"):
            device = {
                "name": name,
                "kind": "capacitor",
                "pin_type": ['cap_plus', 'cap_minus'],
                "pins": tokens[1:3],
                "params": {
                    "cap": tokens[3]
                }
            }
            devices.append(device)
        elif name.startswith("I"):
            device = {
                "name": name,
                "kind": "current_source",
                "pin_type": ['i_plus', 'i_minus'],
                "pins": tokens[1:3],
                "params": {
                    "idc": tokens[3]
                }
            }
            devices.append(device)
    with open(device_message_path, encoding='utf-8', mode='w') as f:
        json.dump(devices, f, indent=4, ensure_ascii=False)
    print(f"器件信息已保存到{device_message_path}")
        
    return devices