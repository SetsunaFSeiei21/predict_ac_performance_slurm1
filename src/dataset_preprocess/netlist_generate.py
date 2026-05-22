import os
import numpy as np
from typing import List, Dict, Any

def generate_pin_level_netlist(device_messages: List[Dict[str, Any]], circuit_dir: str) -> None:
    
    netlist_file_path = os.path.join(circuit_dir, "pin_level_netlist.npy")
    if os.path.exists(netlist_file_path):
        return
    net2pin_dict: Dict[str, List[str]] = {}
    pin_order_lst: List[str] = []
    for message in device_messages:
        device_name = message['name']
        for i in range(len(message['pin_type'])):
            pin_type = message['pin_type'][i]
            cor_net_name = message['pins'][i]
            pin_name = f"{device_name}_{pin_type}"
            pin_order_lst.append(pin_name)
            if cor_net_name not in net2pin_dict:
                net2pin_dict[cor_net_name] = [pin_name]
            else:
                net2pin_dict[cor_net_name].append(pin_name)
    np_pin_level_netlist = np.zeros(shape=(len(pin_order_lst), len(pin_order_lst)))
    all_pin_connect_list = list(net2pin_dict.values())
    pin2index_dict = {pin_name:index for index, pin_name in enumerate(pin_order_lst)}
    for idx, pin_name in enumerate(pin_order_lst):
        tmp_connect_pin_lst = []
        for i in all_pin_connect_list:
            if pin_name in i:
                tmp_connect_pin_lst.extend(i)
        tmp_connect_pin_lst = set(tmp_connect_pin_lst)
        for i in tmp_connect_pin_lst:
            pin_index = pin2index_dict[i]
            np_pin_level_netlist[idx][pin_index] = 1
    np.save(netlist_file_path, np_pin_level_netlist)
            
def generate_device_level_netlist(device_messages: List[Dict[str, Any]], circuit_dir: str):
    
    netlist_file_path = os.path.join(circuit_dir, "device_level_netlist.npy")
    if os.path.exists(netlist_file_path):
        return
    net2device_dict: Dict[str, List[str]] = {}
    device_order_lst = []
    for message in device_messages:
        device_name = message['name']
        device_order_lst.append(device_name)
        for net_name in message['pins']:
            if net_name not in net2device_dict:
                net2device_dict[net_name] = [device_name]
            else:
                if device_name not in net2device_dict[net_name]:
                    net2device_dict[net_name].append(device_name)
    # ignore VDD/GND
    vdd_symbol_lst = ['vdd', 'VDD', 'Vdd']
    gnd_symbol_lst = ['GND', 'gnd', 'gnd!']
    for vdd_symbol in vdd_symbol_lst:
        if vdd_symbol in net2device_dict:
            del net2device_dict[vdd_symbol]
    for gnd_symbol in gnd_symbol_lst:
        if gnd_symbol in net2device_dict:
            del net2device_dict[gnd_symbol]
    np_device_level_netlist = np.zeros((len(device_order_lst), len(device_order_lst)))
    all_connect_lst = list(net2device_dict.values())
    device2idx_dict = {device:idx for idx, device in enumerate(device_order_lst)}
    for idx, device_name in enumerate(device_order_lst):
        tmp_connect_device_lst = []
        for i in all_connect_lst:
            if device_name in i:
                tmp_connect_device_lst.extend(i)
        tmp_connect_device_lst = set(tmp_connect_device_lst)
        for i in tmp_connect_device_lst:
            device_index = device2idx_dict[i]
            np_device_level_netlist[idx][device_index] = 1
    np.save(netlist_file_path, np_device_level_netlist)