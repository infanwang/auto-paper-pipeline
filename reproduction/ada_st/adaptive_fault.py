#!/usr/bin/env python3
"""
ADA-ST改进: LLM智能故障注入 + 时序故障 + 跨层传播

原论文: ADA-ST (2607.16161)
改进内容:
1. 基于规则的智能故障注入(替代随机)
2. 时序故障模型(故障随时间演化)
3. 跨层故障传播模拟
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================
# 1. Fault Models
# ============================================================

class FaultType:
    STUCK_AT_0 = "stuck_at_0"
    STUCK_AT_1 = "stuck_at_1"
    BIT_FLIP = "bit_flip"
    TRANSIENT = "transient"
    PATTERN = "pattern"

class FaultModel:
    """故障模型"""
    
    @staticmethod
    def stuck_at(value, bit_pos=None):
        """固定故障"""
        if bit_pos is not None:
            mask = 1 << bit_pos
            return value | mask if value & mask == 0 else value
        return 0 if value > 0.5 else 1
    
    @staticmethod
    def bit_flip(value, n_flips=1):
        """位翻转故障"""
        for _ in range(n_flips):
            pos = np.random.randint(0, 32)
            value ^= (1 << pos)
        return value
    
    @staticmethod
    def transient(value, prob=0.1):
        """瞬态故障"""
        if np.random.random() < prob:
            return value ^ (1 << np.random.randint(0, 32))
        return value
    
    @staticmethod
    def pattern(values, pattern_type='counter'):
        """模式故障"""
        if pattern_type == 'counter':
            return [(v + 1) % 256 for v in values]
        elif pattern_type == 'shift':
            return [v << 1 for v in values]
        return values


# ============================================================
# 2. Smart Fault Injector (基于规则的智能注入)
# ============================================================

class SmartFaultInjector:
    """改进: 基于规则的智能故障注入"""
    
    def __init__(self, n_layers=8, n_units=64):
        self.n_layers = n_layers
        self.n_units = n_units
        
        # 每层的敏感度
        self.layer_sensitivity = np.ones(n_layers) / n_layers
        
        # 故障传播概率
        self.propagation_prob = np.zeros((n_layers, n_layers))
        for i in range(n_layers-1):
            self.propagation_prob[i, i+1] = 0.3  # 30%概率传播到下一层
    
    def analyze_critical_paths(self, activation_stats):
        """分析关键路径(哪些单元更重要)"""
        criticality = np.zeros((self.n_layers, self.n_units))
        
        for layer_idx in range(self.n_layers):
            if layer_idx in activation_stats:
                stats = activation_stats[layer_idx]
                # 激活幅度大的单元更关键
                criticality[layer_idx] = np.abs(stats.get('mean', np.zeros(self.n_units)))
        
        # 归一化
        criticality = criticality / (criticality.max() + 1e-8)
        return criticality
    
    def smart_inject(self, fault_rate=0.1, criticality=None):
        """智能故障注入: 关键路径低故障率，非关键路径高故障率"""
        faults = []
        
        for layer in range(self.n_layers):
            for unit in range(self.n_units):
                if criticality is not None:
                    # 关键单元降低故障率
                    effective_rate = fault_rate * (1 - criticality[layer, unit])
                else:
                    effective_rate = fault_rate
                
                if np.random.random() < effective_rate:
                    fault_type = np.random.choice([
                        FaultType.STUCK_AT_0, FaultType.STUCK_AT_1,
                        FaultType.BIT_FLIP, FaultType.TRANSIENT
                    ])
                    
                    faults.append({
                        'layer': layer,
                        'unit': unit,
                        'type': fault_type,
                        'severity': 1 - criticality[layer, unit] if criticality is not None else 1.0
                    })
        
        return faults


# ============================================================
# 3. Temporal Fault Simulator (时序故障)
# ============================================================

class TemporalFaultSimulator:
    """改进: 时序故障模型"""
    
    def __init__(self, n_steps=100, degradation_rate=0.01):
        self.n_steps = n_steps
        self.degradation_rate = degradation_rate
        self.fault_history = []
    
    def simulate(self, fault_rate=0.05):
        """模拟时序故障演化"""
        current_health = 1.0  # 健康度
        results = []
        
        for step in range(self.n_steps):
            # 故障率随时间增加(老化效应)
            effective_rate = fault_rate * (1 + self.degradation_rate * step)
            
            # 注入故障
            n_faults = np.random.binomial(100, effective_rate)  # 假设100个单元
            
            # 健康度下降
            current_health *= (1 - n_faults * 0.001)
            current_health = max(current_health, 0.0)
            
            results.append({
                'step': step,
                'n_faults': n_faults,
                'health': current_health,
                'effective_rate': effective_rate
            })
        
        self.fault_history = results
        return results
    
    def detect_anomalies(self, window=10):
        """检测故障异常(超过阈值)"""
        if len(self.fault_history) < window:
            return []
        
        anomalies = []
        for i in range(window, len(self.fault_history)):
            recent = [r['n_faults'] for r in self.fault_history[i-window:i]]
            mean = np.mean(recent)
            std = np.std(recent)
            current = self.fault_history[i]['n_faults']
            
            if current > mean + 2 * std:
                anomalies.append(i)
        
        return anomalies


# ============================================================
# 4. Cross-Layer Propagation (跨层传播)
# ============================================================

class CrossLayerPropagation:
    """改进: 跨层故障传播"""
    
    def __init__(self, n_layers=8):
        self.n_layers = n_layers
        
        # 传播矩阵: propagation_prob[i][j] = 故障从层i传播到层j的概率
        self.propagation = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(i+1, min(i+3, n_layers)):
                self.propagation[i, j] = 0.2 / (j - i)  # 距离越远概率越低
    
    def propagate(self, initial_faults: List[Dict]) -> List[Dict]:
        """模拟故障传播"""
        all_faults = list(initial_faults)
        current_faults = list(initial_faults)
        
        for _ in range(3):  # 最多传播3轮
            new_faults = []
            for fault in current_faults:
                src_layer = fault['layer']
                for dst_layer in range(src_layer+1, self.n_layers):
                    if np.random.random() < self.propagation[src_layer, dst_layer]:
                        new_faults.append({
                            'layer': dst_layer,
                            'unit': fault['unit'],
                            'type': fault['type'],
                            'severity': fault['severity'] * 0.5,  # 严重性衰减
                            'propagated_from': src_layer
                        })
            all_faults.extend(new_faults)
            current_faults = new_faults
        
        return all_faults


# ============================================================
# 5. Benchmark
# ============================================================

def main():
    print("="*60)
    print("ADA-ST改进: 智能故障注入 + 时序故障 + 跨层传播")
    print("="*60)
    
    # 1. 智能故障注入
    print("\n[1] 智能故障注入:")
    injector = SmartFaultInjector(n_layers=8, n_units=64)
    
    # 模拟激活统计
    activation_stats = {i: {'mean': np.random.randn(64)} for i in range(8)}
    criticality = injector.analyze_critical_paths(activation_stats)
    
    # 对比: 随机 vs 智能
    random_faults = []
    for _ in range(100):
        faults = injector.smart_inject(fault_rate=0.1, criticality=None)
        random_faults.append(len(faults))
    
    smart_faults = []
    for _ in range(100):
        faults = injector.smart_inject(fault_rate=0.1, criticality=criticality)
        smart_faults.append(len(faults))
    
    print(f"  随机注入: {np.mean(random_faults):.1f} faults/run")
    print(f"  智能注入: {np.mean(smart_faults):.1f} faults/run")
    print(f"  关键路径保护: {np.mean(random_faults)/np.mean(smart_faults):.2f}x减少")
    
    # 2. 时序故障
    print("\n[2] 时序故障模拟:")
    temporal = TemporalFaultSimulator(n_steps=100, degradation_rate=0.02)
    results = temporal.simulate(fault_rate=0.05)
    anomalies = temporal.detect_anomalies(window=10)
    
    print(f"  初始健康度: 1.000")
    print(f"  最终健康度: {results[-1]['health']:.3f}")
    print(f"  异常检测: {len(anomalies)}个")
    
    # 3. 跨层传播
    print("\n[3] 跨层故障传播:")
    propagator = CrossLayerPropagation(n_layers=8)
    initial = [{'layer': 2, 'unit': 10, 'type': 'bit_flip', 'severity': 1.0}]
    propagated = propagator.propagate(initial)
    
    layers_affected = set(f['layer'] for f in propagated)
    print(f"  初始故障: 1个 (层2)")
    print(f"  传播后: {len(propagated)}个故障")
    print(f"  影响层: {sorted(layers_affected)}")
    
    # 4. 改进汇总
    print("\n[4] 改进汇总:")
    print("  原版ADA-ST: 随机故障注入")
    print("  改进ADA-ST:")
    print("    - 智能注入(关键路径保护)")
    print("    - 时序故障(老化效应)")
    print("    - 跨层传播(故障扩散)")
    
    # 保存结果
    results_data = {
        'smart_inject': {'random': float(np.mean(random_faults)), 'smart': float(np.mean(smart_faults))},
        'temporal': {'initial_health': 1.0, 'final_health': float(results[-1]['health']), 'anomalies': len(anomalies)},
        'propagation': {'initial': 1, 'propagated': len(propagated), 'layers_affected': sorted(layers_affected)}
    }
    
    out = Path('/root/git/mimo/paper-pipeline/reproduction/ada_st/results.json')
    out.write_text(json.dumps(results_data, indent=2))
    print(f"\n结果: {out}")

if __name__ == "__main__":
    main()
