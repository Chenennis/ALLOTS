"""
Configuration Validator  
配置验证工具 - 验证CSV文件和环境配置的合理性
"""
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

test_dir = str(Path(__file__).parent)
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    from test_env_config import TestEnvironmentConfig
except ImportError:
    from test.test_env_config import TestEnvironmentConfig


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_csv_files(self, 
                          managers_csv: str,
                          users_csv: str,
                          devices_csv: str) -> Tuple[bool, List[str], List[str]]:
        """验证CSV文件
        
        Args:
            managers_csv: Manager配置CSV文件路径
            users_csv: 用户配置CSV文件路径
            devices_csv: 设备配置CSV文件路径
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # 读取文件
        try:
            managers = self._read_csv(managers_csv)
            users = self._read_csv(users_csv)
            devices = self._read_csv(devices_csv)
        except Exception as e:
            self.errors.append(f"Failed to read CSV files: {e}")
            return (False, self.errors, self.warnings)
        
        # 验证Manager配置
        self._validate_managers(managers)
        
        # 验证用户配置
        self._validate_users(users, managers)
        
        # 验证设备配置
        self._validate_devices(devices, users)
        
        # 统计分析
        self._analyze_distribution(managers, users, devices)
        
        is_valid = len(self.errors) == 0
        return (is_valid, self.errors, self.warnings)
    
    def validate_config_object(self, config: TestEnvironmentConfig) -> Tuple[bool, List[str], List[str]]:
        """验证配置对象
        
        Args:
            config: TestEnvironmentConfig对象
        
        Returns:
            (is_valid, errors, warnings)
        """
        is_valid, errors = config.validate()
        return (is_valid, errors, [])
    
    def _read_csv(self, filepath: str) -> List[Dict]:
        """读取CSV文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    
    def _validate_managers(self, managers: List[Dict]):
        """验证Manager配置"""
        manager_ids = set()
        
        for manager in managers:
            mid = manager.get('manager_id', '')
            
            # 检查必需字段
            required = ['manager_id', 'location_x', 'location_y', 'coverage_area', 'district_type', 'user_count']
            for field in required:
                if field not in manager or not manager[field]:
                    self.errors.append(f"Manager {mid}: missing required field '{field}'")
            
            # 检查ID唯一性
            if mid in manager_ids:
                self.errors.append(f"Duplicate manager_id: {mid}")
            manager_ids.add(mid)
            
            # 检查数值合理性
            try:
                if float(manager.get('coverage_area', 0)) <= 0:
                    self.errors.append(f"Manager {mid}: invalid coverage_area")
            except ValueError:
                self.errors.append(f"Manager {mid}: coverage_area must be a number")
            
            try:
                if int(manager.get('user_count', 0)) <= 0:
                    self.errors.append(f"Manager {mid}: invalid user_count")
            except ValueError:
                self.errors.append(f"Manager {mid}: user_count must be an integer")
    
    def _validate_users(self, users: List[Dict], managers: List[Dict]):
        """验证用户配置"""
        user_ids = set()
        manager_ids = {m['manager_id'] for m in managers}
        manager_user_counts = Counter()
        
        for user in users:
            uid = user.get('user_id', '')
            mid = user.get('manager_id', '')
            
            # 检查必需字段
            required = ['user_id', 'manager_id', 'location_x', 'location_y', 'user_type',
                       'economic_pref', 'comfort_pref', 'environmental_pref']
            for field in required:
                if field not in user or not user[field]:
                    self.errors.append(f"User {uid}: missing required field '{field}'")
            
            # 检查ID唯一性
            if uid in user_ids:
                self.errors.append(f"Duplicate user_id: {uid}")
            user_ids.add(uid)
            
            # 检查manager_id有效性
            if mid not in manager_ids:
                self.errors.append(f"User {uid}: references unknown manager_id '{mid}'")
            else:
                manager_user_counts[mid] += 1
            
            # 检查偏好总和
            try:
                eco = float(user.get('economic_pref', 0))
                com = float(user.get('comfort_pref', 0))
                env = float(user.get('environmental_pref', 0))
                total = eco + com + env
                
                if not (0.99 <= total <= 1.01):
                    self.errors.append(f"User {uid}: preferences sum to {total:.3f}, should be 1.0")
                
                if eco < 0 or com < 0 or env < 0:
                    self.errors.append(f"User {uid}: preferences must be non-negative")
            except ValueError:
                self.errors.append(f"User {uid}: preferences must be numbers")
            
            # 检查用户类型
            valid_types = ['prosumer', 'consumer', 'producer']
            if user.get('user_type') not in valid_types:
                self.warnings.append(f"User {uid}: unknown user_type '{user.get('user_type')}'")
        
        # 检查实际用户数量与声明是否匹配
        for manager in managers:
            mid = manager['manager_id']
            declared = int(manager.get('user_count', 0))
            actual = manager_user_counts.get(mid, 0)
            if declared != actual:
                self.warnings.append(f"Manager {mid}: declared {declared} users but has {actual} users")
    
    def _validate_devices(self, devices: List[Dict], users: List[Dict]):
        """验证设备配置"""
        device_ids = set()
        user_ids = {u['user_id'] for u in users}
        user_device_counts = Counter()
        device_type_counts = Counter()
        
        for device in devices:
            did = device.get('device_id', '')
            uid = device.get('user_id', '')
            dtype = device.get('device_type', '')
            
            # 检查必需字段
            required = ['device_id', 'user_id', 'device_type', 'capacity', 'max_power',
                       'efficiency', 'initial_state', 'param1', 'param2', 'param3',
                       'can_interrupt', 'priority']
            for field in required:
                if field not in device or device[field] == '':
                    self.errors.append(f"Device {did}: missing required field '{field}'")
            
            # 检查ID唯一性
            if did in device_ids:
                self.errors.append(f"Duplicate device_id: {did}")
            device_ids.add(did)
            
            # 检查user_id有效性
            if uid not in user_ids:
                self.errors.append(f"Device {did}: references unknown user_id '{uid}'")
            else:
                user_device_counts[uid] += 1
            
            # 统计设备类型
            device_type_counts[dtype] += 1
            
            # 检查设备类型
            valid_types = ['battery', 'ev', 'heat_pump', 'pv', 'dishwasher']
            if dtype not in valid_types:
                self.errors.append(f"Device {did}: unknown device_type '{dtype}'")
            
            # 检查数值合理性
            try:
                capacity = float(device.get('capacity', 0))
                max_power = float(device.get('max_power', 0))
                efficiency = float(device.get('efficiency', 0))
                
                if dtype in ['battery', 'ev'] and capacity <= 0:
                    self.errors.append(f"Device {did}: {dtype} must have positive capacity")
                
                if max_power < 0:
                    self.errors.append(f"Device {did}: max_power cannot be negative")
                
                if not (0 < efficiency <= 1.0):
                    # Heat pump的COP可以大于1
                    if dtype != 'heat_pump':
                        self.errors.append(f"Device {did}: efficiency must be between 0 and 1")
                    elif efficiency < 1.0 or efficiency > 10.0:
                        self.warnings.append(f"Device {did}: unusual heat pump COP: {efficiency}")
            except ValueError:
                self.errors.append(f"Device {did}: numeric fields must be numbers")
        
        # 检查设备分布
        total_users = len(user_ids)
        if total_users > 0:
            for dtype, count in device_type_counts.items():
                percentage = (count / total_users) * 100
                if dtype == 'dishwasher' and percentage < 90:
                    self.warnings.append(f"Dishwasher coverage is only {percentage:.0f}%, recommended 100%")
                elif dtype == 'heat_pump' and percentage < 90:
                    self.warnings.append(f"Heat pump coverage is only {percentage:.0f}%, recommended 100%")
    
    def _analyze_distribution(self, managers: List[Dict], users: List[Dict], devices: List[Dict]):
        """分析配置分布"""
        total_users = len(users)
        total_devices = len(devices)
        
        device_types = Counter(d.get('device_type', '') for d in devices)
        
        print(f"\n=== Configuration Summary ===")
        print(f"Managers: {len(managers)}")
        print(f"Users: {total_users}")
        print(f"Devices: {total_devices}")
        print(f"\nDevice Distribution:")
        for dtype, count in sorted(device_types.items()):
            percentage = (count / total_users) * 100 if total_users > 0 else 0
            print(f"  - {dtype}: {count} ({percentage:.0f}%)")


def validate_csv_files(managers_csv: str, users_csv: str, devices_csv: str):
    """验证CSV文件的便捷函数
    
    Args:
        managers_csv: Manager配置CSV文件路径
        users_csv: 用户配置CSV文件路径
        devices_csv: 设备配置CSV文件路径
    """
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_csv_files(
        managers_csv, users_csv, devices_csv
    )
    
    if is_valid:
        print("\n[OK] Configuration is VALID!")
    else:
        print("\n[ERROR] Configuration has errors:")
        for error in errors:
            print(f"  ❌ {error}")
    
    if warnings:
        print("\n[WARNING] Configuration warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    return is_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate FlexOffer test environment configuration")
    parser.add_argument("--managers", required=True, help="Path to managers CSV file")
    parser.add_argument("--users", required=True, help="Path to users CSV file")
    parser.add_argument("--devices", required=True, help="Path to devices CSV file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Configuration Validator")
    print("="*70)
    
    is_valid = validate_csv_files(args.managers, args.users, args.devices)
    
    sys.exit(0 if is_valid else 1)
