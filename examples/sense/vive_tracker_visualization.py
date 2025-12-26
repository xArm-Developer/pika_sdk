#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vive Tracker模块 - 获取设备的位姿数据并用Pygame实时可视化
此示例是为了用户更好的观察设备定位是否有漂移
"""

import sys
import time
import os
import logging
import numpy as np
import pygame
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('vive_tracker_visualization')

# --- Pygame 可视化类 ---
class PygameVisualizer:
    def __init__(self, width=800, height=600):
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Vive Tracker Pose Visualization")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.running = True
        except Exception as e:
            logger.error(f"Pygame初始化失败: {e}")
            self.running = False
            
        # 1. 新增缩放/相机参数
        self.camera_distance = 5.0  # 离z=0平面的距离
        self.scale_factor = 2000.0   # 投影缩放因子，用于缩放视图
        self.zoom_step = 100.0       # 每次缩放的调整量
        
    def project_3d_to_2d(self, point_3d):
        """将3D点投影到2D屏幕"""
        x, y, z = point_3d
        
        # 避免除以零或负值
        depth = z + self.camera_distance
        if depth <= 0.1:
            depth = 0.1 

        # 使用缩放因子控制视图大小
        factor = self.scale_factor / depth 
        x_2d = 400 + x * factor
        y_2d = 300 - y * factor 
        return (int(x_2d), int(y_2d))
    
    def quaternion_to_rotation_matrix(self, quaternion):
        """将 [x, y, z, w] 四元数转换为 3x3 旋转矩阵"""
        q1, q2, q3, q0 = quaternion # 对应 (x, y, z, w)
        
        R = np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])
        return R

    def draw_coordinate_frame(self, position, quaternion, color, length, name=""):
        """
        绘制一个坐标系
        :param position: 坐标系原点 [x, y, z]
        :param quaternion: 坐标系姿态 [x, y, z, w]
        :param color: 轴颜色 (R, G, B)
        :param length: 轴的长度
        :param name: 坐标系名称，用于显示
        """
        # 四元数转旋转矩阵
        R = self.quaternion_to_rotation_matrix(quaternion)
        
        # 坐标系轴
        pos_np = np.array(position)
        x_end = pos_np + R[:, 0] * length
        y_end = pos_np + R[:, 1] * length
        z_end = pos_np + R[:, 2] * length
        
        # 投影到2D
        pos_2d = self.project_3d_to_2d(pos_np)
        x_end_2d = self.project_3d_to_2d(x_end)
        y_end_2d = self.project_3d_to_2d(y_end)
        z_end_2d = self.project_3d_to_2d(z_end)
        
        # 绘制轴线 (使用传入的颜色)
        pygame.draw.line(self.screen, (255, 0, 0), pos_2d, x_end_2d, 3)  # X轴-红
        pygame.draw.line(self.screen, (0, 255, 0), pos_2d, y_end_2d, 3)  # Y轴-绿
        pygame.draw.line(self.screen, (0, 0, 255), pos_2d, z_end_2d, 3)  # Z轴-蓝

        # 绘制原点
        pygame.draw.circle(self.screen, color, pos_2d, 5)
        
        # 显示坐标系名称
        if name:
            text_surface = self.font.render(name, True, color)
            self.screen.blit(text_surface, (pos_2d[0] + 10, pos_2d[1] - 10))

    def handle_input(self):
        """处理用户输入，包括退出和缩放"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return 
            
            if event.type == pygame.KEYDOWN:
                # 放大 (W键 或 Up Arrow)
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.scale_factor += self.zoom_step
                    logger.info(f"缩放: 放大到 {self.scale_factor:.2f}")
                # 缩小 (S键 或 Down Arrow)
                if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    if self.scale_factor > self.zoom_step:
                        self.scale_factor -= self.zoom_step
                        logger.info(f"缩放: 缩小到 {self.scale_factor:.2f}")

    def update(self, tracker_position, tracker_rotation):
        """更新显示，绘制 Base Link 和 Tracker 坐标系"""
        self.handle_input()
        if not self.running:
            return 

        self.screen.fill((0, 0, 0)) # 黑色背景
        
        # --- 1. 绘制 Base Link 坐标系 (零点) ---
        # 零点位置，无旋转
        base_pos = [0.0, 0.0, 0.0]
        base_quat = [0.0, 0.0, 0.0, 1.0] 
        self.draw_coordinate_frame(
            position=base_pos, 
            quaternion=base_quat, 
            color=(100, 100, 255), # 浅蓝
            length=0.7, # 轴长度 0.7m
            name="base_Link"
        )
        
        # --- 2. 绘制 Tracker 坐标系 (相对 Base Link) ---
        # tracker_position 和 tracker_rotation 本身就是相对于 Base Link (世界坐标系) 的位姿
        self.draw_coordinate_frame(
            position=tracker_position, 
            quaternion=tracker_rotation, 
            color=(255, 255, 0), # 黄色
            length=0.5, # 轴长度 0.5m
            name="tracker_link"
        )
        
        # 绘制 Base Link 到 Tracker 的连线 (更清晰地表示相对位置)
        pos_2d_base = self.project_3d_to_2d(base_pos)
        pos_2d_tracker = self.project_3d_to_2d(tracker_position)
        pygame.draw.line(self.screen, (150, 150, 150), pos_2d_base, pos_2d_tracker, 1)

        # 显示位姿信息
        info_text = f"Position: ({tracker_position[0]:.4f}, {tracker_position[1]:.4f}, {tracker_position[2]:.4f}) (m)"
        quat_text = f"Rotation (x, y, z, w): ({tracker_rotation[0]:.4f}, {tracker_rotation[1]:.4f}, {tracker_rotation[2]:.4f}, {tracker_rotation[3]:.4f})"
        zoom_text = f"Zoom/Scale: {self.scale_factor/200.0:.2f}x (W/S or Up/Down)"

        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        quat_surface = self.font.render(quat_text, True, (255, 255, 255))
        self.screen.blit(quat_surface, (10, 40))

        zoom_surface = self.font.render(zoom_text, True, (200, 200, 200))
        self.screen.blit(zoom_surface, (10, 70))
        
        pygame.display.flip()
        self.clock.tick(60) # 保持60FPS
        
    def is_running(self):
        return self.running

# --- Vive Tracker 逻辑 ---
try:
    from pika.sense import Sense
    
    def run_visualization():
        """测试获取WM0设备的位姿数据并可视化"""
        viz = PygameVisualizer()
        if not viz.is_running():
            logger.error("Pygame可视化环境未准备好，退出。")
            return False

        sense = Sense()
        logger.info("连接Sense设备...")
        if not sense.connect():
            logger.error("连接Sense设备失败")
            pygame.quit() 
            return False
        
        try:
            tracker = sense.get_vive_tracker()
            if not tracker:
                logger.error("获取Vive Tracker对象失败，请确保已安装pysurvive库")
                pygame.quit()
                return False
            
            logger.info("等待设备初始化完成...")
            time.sleep(2.0)
            
            target_device = "WM0"
            max_retries = 10
            retry_count = 0
            # 设备连接检查逻辑... 
            for retry_count in range(max_retries):
                devices = sense.get_tracker_devices()
                if target_device in devices:
                    logger.info(f"成功检测到{target_device}设备！")
                    break
                else:
                    logger.info(f"未检测到{target_device}设备，等待并重试 ({retry_count+1}/{max_retries})...")
                time.sleep(1.0)
            else:
                logger.warning(f"经过多次尝试，仍未检测到{target_device}设备")
                pygame.quit()
                return False
            
            logger.info(f"开始获取{target_device}设备的位姿数据并更新可视化 (W/S或↑/↓键控制缩放，按X关闭窗口结束)...")
            
            # 循环获取数据并更新可视化
            while viz.is_running(): 
                pose = sense.get_pose(target_device)
                
                if pose:
                    position = pose.position  # [x, y, z]
                    rotation = pose.rotation  # [x, y, z， w] 四元数
                    viz.update(position, rotation)
                else:
                    logger.warning(f"未能获取{target_device}的位姿数据...")
                    # 至少调用一次 update 来处理用户输入事件
                    viz.update([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
                
                time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"获取过程中发生错误: {e}")
        finally:
            logger.info("断开Sense设备连接...")
            sense.disconnect()
            pygame.quit()
    
    if __name__ == "__main__":
        run_visualization()
        
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error("请确保已安装所有必要的依赖：pysurvive 库、pika.sense 库 和 pygame 库")