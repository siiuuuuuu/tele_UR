"""
这是一个直接设置手指角度的例程
"""
import numpy as np

from InspireHandControl_V1 import InspireHand
import time
import socket
import os
import time
import struct

class ManusTeleoperationController:
    """MANUS手套到Inspire灵巧手的遥操作控制器"""
    
    def __init__(self, serial_port="/dev/ttyUSB0", baudrate=115200, 
                 manus_port=8888, control_threshold=10, scale_factor=15):
        """
        初始化遥操作控制器
        
        Args:
            serial_port: 灵巧手串口 (默认: /dev/ttyUSB0)
            baudrate: 串口波特率 (默认: 115200)
            manus_port: MANUS数据端口 (默认: 8888)
            control_threshold: 控制阈值 (默认: 10)
            scale_factor: 数据缩放因子 (默认: 15)
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.manus_port = manus_port
        self.control_threshold = control_threshold
        self.scale_factor = scale_factor
        
        # 初始化灵巧手
        self.right_hand = None
        self.left_hand_connect = False
        self.right_hand_connect = False
        
        # 数据缓存
        self.right_hand_data0 = None
        self.set_data = None
        
        # 网络连接
        self.server = None
        self.connection = None
        self.address = None
        
        print(f"🤖 MANUS遥操作控制器初始化完成")
        print(f"📡 串口: {serial_port} @ {baudrate} bps")
        print(f"🌐 MANUS端口: {manus_port}")
    
    def initialize_hand(self):
        """初始化灵巧手连接"""
        try:
            print("🔄 正在连接灵巧手...")
            self.right_hand = InspireHand(self.serial_port, self.baudrate)
            time.sleep(1)
            self.right_hand.reset()
            time.sleep(1)
            print(f"✅ 灵巧手连接成功: {self.right_hand.num2str(0x70)}")
            
            # 设置灵巧手参数
            self.right_hand.setpower(500, 500, 500, 500, 500)  # 力控参数，其采用导纳控制，即便给定目标关节角度，实际位置也会根据外力有所偏移
            # 靠力传感器得到接触力，去调节关节位置（达到一个位置使得接触力达到这个目标值。而不是硬到给定关节角度）
            self.right_hand.setspeed(300, 300, 300, 300, 300)
            print("⚙️ 灵巧手参数设置完成")
            
        except Exception as e:
            print(f"❌ 灵巧手连接失败: {e}")
            raise
    
    def initialize_manus_connection(self):
        """初始化MANUS网络连接"""
        try:
            print("🔄 正在建立MANUS连接...")
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind(("localhost", self.manus_port))
            print("✅ MANUS服务器绑定成功")
            
            self.server.listen(0)
            print("👂 正在监听MANUS数据...")
            
            self.connection, self.address = self.server.accept()
            print(f"🔗 MANUS客户端连接成功: {self.address}")
            
        except Exception as e:
            print(f"❌ MANUS连接失败: {e}")
            raise
    
    def receive_manus_data(self):
        """接收MANUS手套数据"""
        try:
            data = []
            for i in range(9):
                recv_bytes = self.connection.recv(8)  # 每次精确收8字节
                if not recv_bytes:
                    return None
                float_str = recv_bytes.decode('ascii')
                value = float(float_str)
                data.append(value)
            return data
        except Exception as e:
            print(f"❌ 数据接收失败: {e}")
            return None
    
    def process_hand_calibration(self, data):
        """处理手部校准"""
        if not self.right_hand_connect and data[0] == 1002 and data[1] != 0:
            self.right_hand_connect = True
            self.right_hand_data0 = np.array(data)  # 记录初始位置
            self.set_data = self.right_hand_data0 - self.right_hand_data0  # 初始化SET为0
            print("🎯 右手校准完成!")
            return True
        return False
    
    def calculate_finger_angles(self, data):
        """计算手指角度"""
        if not self.right_hand_connect or data[0] != 1002:
            return None
        
        # 计算相对变化
        diff = np.array(data) - self.set_data
        scaled_data = np.array(data) * self.scale_factor
        self.set_data = np.array(scaled_data) - self.right_hand_data0  # 更新SET为当前相对位置
        
        # 检查控制阈值
        if np.max(np.abs(diff)) < self.control_threshold:
            return None
        
        return self.set_data
    
    def send_hand_command(self, angles):
        """发送手部控制命令"""
        if angles is None:
            return
        
        try:
            pinky_angle = int(max(0, min(1000, 1000 - angles[8])))
            ring_angle = int(max(0, min(1000, 1000 - angles[7])))
            middle_angle = int(max(0, min(1000, 1000 - angles[6])))
            index_angle = int(max(0, min(1000, 1000 - angles[4])))
            thumb_angle_2 = int(min(1000, max(0, 0 + 1.5 * angles[2])))
            thumb_angle = int(min(1000, max(0, 1000 - 1.5 * angles[1])))
            
            # 映射到灵巧手角度范围并发送命令
            self.right_hand.setangle(
                pinky_angle,      
                ring_angle,      
                middle_angle,     
                index_angle,       
                thumb_angle_2,     #大拇指弯曲
                thumb_angle   #大拇指侧摆    
            )
        except Exception as e:
            print(f"❌ 手部命令发送失败: {e}")
    
    def run_teleoperation_loop(self):
        loop_count = 0
        try:
            while True:
                time_start=time.time()
                # 接收MANUS数据
                data = self.receive_manus_data()
                if data is None:
                    raise ConnectionError("MANUS数据接收中断")
                    
                
                #print(f"📡 原始数据: {[f'{d:.1f}' for d in data]}")
                
                # 处理手部校准
                self.process_hand_calibration(data)
                
                # 计算手指角度
                angles = self.calculate_finger_angles(data)
                # 发送控制命令
                self.send_hand_command(angles)
                
                loop_count += 1
                #if loop_count % 100 == 0:
                    #print(f"🔄 运行中... 循环次数: {loop_count}")
                end_time=time.time()
                time.sleep(max(0, 1/50 - (end_time - time_start)))  # 60Hz读取频率
                #c++端是30ms更新数据，所以一般需要45hz以上不让缓冲区的数据累积导致拿到的是旧数据造成延迟
                
        except KeyboardInterrupt:
            print("\n⏹️  用户中断遥操作")
        except ConnectionError as e:
            print(f"❌ 连接错误: {e}")
        except Exception as e:
            print(f"❌ 遥操作异常: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 正在清理资源...")
        
        try:
            if self.connection:
                self.connection.close()
                print("🔗 网络连接已关闭")
            
            if self.server:
                self.server.close()
                print("🌐 服务器已关闭")
            
            if self.right_hand:
                self.right_hand.reset()
                self.right_hand.close()
                print("🤖 灵巧手已重置并关闭")
                
        except Exception as e:
            print(f"⚠️  清理过程中出错: {e}")
        
        print("✅ 清理完成")
    def prepare(self):
        """准备遥操作系统"""
        try:
            print("🔧 准备遥操作系统...")
            self.initialize_hand()
            self.initialize_manus_connection()
        except Exception as e:
            print(f"❌ 准备遥操作系统失败: {e}")
            self.cleanup()
            return False
        return True

    def start(self):
        """启动遥操作系统"""
        try:
            # 运行主循环
            self.run_teleoperation_loop()
            
        except Exception as e:
            print(f"❌ 系统启动失败: {e}")
            self.cleanup()

if __name__ == '__main__':
    
    # 创建控制器实例
    controller = ManusTeleoperationController(
        serial_port="/dev/ttyUSB0",  # 或者 "/dev/ttyUSB1"
        baudrate=115200,
        manus_port=8888,
        control_threshold=10,
        scale_factor=15
    )
    flag=controller.prepare()
    # 启动遥操作系统
    if flag:
        controller.start()