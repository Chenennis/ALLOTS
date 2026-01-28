"""
生成10Manager测试环境的CSV数据文件
用户分布：(6,10,8,12,8,8,10,6,10,12) = 78个用户
设备比例：洗碗机100%，EV 40-60%，PV 20-40%，Battery 80-100%
"""
import csv
import random
import os

# 设置随机种子保证可重复性
random.seed(42)

# 确保data目录存在
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
print(f"Data directory: {data_dir}")

# 定义用户分布
user_counts = [6, 10, 8, 12, 8, 8, 10, 6, 10, 12]  # 总共90个用户
total_users = sum(user_counts)
print(f"Total users: {total_users}")

# Manager配置
managers = []
for i in range(1, 11):
    managers.append({
        'manager_id': f'manager_{i}',
        'location_x': round(3.5 + (i-1) * 3.0 + ((i-1)%3)*0.5, 1),
        'location_y': round(2.8 + (i-1) * 2.5 + ((i-1)%2)*0.3, 1),
        'coverage_area': round(1.5 + ((i-1)%4)*0.3, 1),
        'district_type': ['residential', 'mixed', 'commercial'][(i-1)%3],
        'user_count': user_counts[i-1]
    })

# 保存Manager CSV
managers_file = os.path.join(data_dir, '10manager_managers.csv')
with open(managers_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['manager_id', 'location_x', 'location_y', 'coverage_area', 'district_type', 'user_count'])
    writer.writeheader()
    writer.writerows(managers)

print(f'[OK] Created 10manager_managers.csv with {len(managers)} managers')

# 生成用户（按照新的分布）
users = []
for m in range(1, 11):
    user_count = user_counts[m-1]
    base_x = managers[m-1]['location_x']
    base_y = managers[m-1]['location_y']
    
    for u in range(1, user_count + 1):
        user_type = random.choice(['prosumer', 'consumer', 'consumer', 'consumer'])  # 25% prosumer
        
        # 生成归一化的偏好
        prefs = [random.uniform(0.3, 0.5), random.uniform(0.3, 0.4), 0.2]
        total = sum(prefs)
        prefs = [round(p/total, 2) for p in prefs]
        prefs[2] = round(1.0 - prefs[0] - prefs[1], 2)  # 确保总和为1
        
        users.append({
            'user_id': f'user_{m}_{u}',
            'manager_id': f'manager_{m}',
            'location_x': round(base_x + random.uniform(-0.5, 0.5), 1),
            'location_y': round(base_y + random.uniform(-0.5, 0.5), 1),
            'user_type': user_type,
            'economic_pref': prefs[0],
            'comfort_pref': prefs[1],
            'environmental_pref': prefs[2]
        })

# 保存用户CSV
users_file = os.path.join(data_dir, '10manager_users.csv')
with open(users_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['user_id', 'manager_id', 'location_x', 'location_y', 'user_type', 'economic_pref', 'comfort_pref', 'environmental_pref'])
    writer.writeheader()
    writer.writerows(users)

print(f'[OK] Created 10manager_users.csv with {len(users)} users')

# 生成设备
all_user_ids = [u['user_id'] for u in users]
devices = []

# 1. 洗碗机 - 100%覆盖（78个）
for user_id in all_user_ids:
    device_id = user_id.replace('user', 'dishwasher')
    devices.append({
        'device_id': device_id,
        'user_id': user_id,
        'device_type': 'dishwasher',
        'capacity': round(random.uniform(2.7, 3.2), 1),
        'max_power': round(random.uniform(1.9, 2.4), 1),
        'efficiency': round(random.uniform(0.87, 0.92), 2),
        'initial_state': 0.0,
        'param1': round(random.uniform(3.2, 3.8), 1),  # operation_hours
        'param2': round(random.uniform(0.5, 0.7), 1),  # min_start_delay
        'param3': round(random.uniform(6.5, 7.5), 1),  # max_start_delay
        'can_interrupt': 0,
        'priority': random.randint(2, 4)
    })

# 2. Heat Pump - 100%覆盖（78个）
for user_id in all_user_ids:
    device_id = user_id.replace('user', 'heatpump')
    devices.append({
        'device_id': device_id,
        'user_id': user_id,
        'device_type': 'heat_pump',
        'capacity': 0.0,
        'max_power': round(random.uniform(3.8, 5.5), 1),
        'efficiency': round(random.uniform(3.5, 4.4), 1),  # COP
        'initial_state': round(random.uniform(20.0, 21.5), 1),
        'param1': 18.0,  # temp_min
        'param2': 26.0,  # temp_max
        'param3': round(random.uniform(0.07, 0.11), 2),  # heat_loss_coef
        'can_interrupt': 1,
        'priority': 4
    })

# 3. Battery - 85%覆盖（约66个，在80-100%范围内）
battery_count = int(total_users * 0.85)  # 78 * 0.85 = 66.3 -> 66个
battery_users = random.sample(all_user_ids, battery_count)
for user_id in battery_users:
    device_id = user_id.replace('user', 'battery')
    capacity = round(random.uniform(8.3, 12.0), 1)
    devices.append({
        'device_id': device_id,
        'user_id': user_id,
        'device_type': 'battery',
        'capacity': capacity,
        'max_power': round(capacity * random.uniform(0.45, 0.55), 1),
        'efficiency': round(random.uniform(0.93, 0.96), 2),
        'initial_state': round(random.uniform(0.5, 0.6), 2),
        'param1': 0.1,  # soc_min
        'param2': 0.9,  # soc_max
        'param3': capacity * 1000,  # capacity_wh
        'can_interrupt': 1,
        'priority': 3
    })

# 4. EV - 50%覆盖（约39个，在40-60%范围内）
ev_count = int(total_users * 0.50)  # 78 * 0.50 = 39个
ev_users = random.sample(all_user_ids, ev_count)
for user_id in ev_users:
    device_id = user_id.replace('user', 'ev')
    capacity = round(random.uniform(50.0, 70.0), 1)
    devices.append({
        'device_id': device_id,
        'user_id': user_id,
        'device_type': 'ev',
        'capacity': capacity,
        'max_power': round(random.uniform(6.0, 8.0), 1),
        'efficiency': round(random.uniform(0.88, 0.92), 2),
        'initial_state': round(random.uniform(0.4, 0.6), 2),
        'param1': 0.2,  # soc_min
        'param2': 0.95,  # soc_max
        'param3': round(random.uniform(18.0, 20.5), 1),  # departure_hour
        'can_interrupt': 1,
        'priority': 2
    })

# 5. PV - 30%覆盖（约23个，在20-40%范围内）
pv_count = int(total_users * 0.30)  # 78 * 0.30 = 23.4 -> 23个
pv_users = random.sample(all_user_ids, pv_count)
for user_id in pv_users:
    device_id = user_id.replace('user', 'pv')
    devices.append({
        'device_id': device_id,
        'user_id': user_id,
        'device_type': 'pv',
        'capacity': 0.0,
        'max_power': round(random.uniform(5.2, 6.8), 1),
        'efficiency': round(random.uniform(0.17, 0.22), 2),
        'initial_state': 0.0,
        'param1': round(random.uniform(27.0, 35.0), 1),  # tilt_angle
        'param2': round(random.uniform(165.0, 190.0), 1),  # azimuth_angle
        'param3': round(random.uniform(26.0, 38.0), 1),  # area
        'can_interrupt': 0,
        'priority': 1
    })

# 按device_id排序
devices.sort(key=lambda x: x['device_id'])

# 保存设备CSV
devices_file = os.path.join(data_dir, '10manager_devices.csv')
with open(devices_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['device_id', 'user_id', 'device_type', 'capacity', 'max_power', 'efficiency', 'initial_state', 'param1', 'param2', 'param3', 'can_interrupt', 'priority']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(devices)

print(f'[OK] Created 10manager_devices.csv with {len(devices)} devices')
print(f'   - Dishwashers: {total_users} (100%)')
print(f'   - Heat Pumps: {total_users} (100%)')
print(f'   - Batteries: {battery_count} ({battery_count/total_users*100:.1f}%)')
print(f'   - EVs: {ev_count} ({ev_count/total_users*100:.1f}%)')
print(f'   - PVs: {pv_count} ({pv_count/total_users*100:.1f}%)')
print(f'   - Total: {len(devices)} devices')
print('\n[OK] All 10Manager CSV files generated successfully!')
