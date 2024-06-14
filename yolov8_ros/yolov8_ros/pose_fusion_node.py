import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from yolov8_msgs.msg import DetectionArray
import numpy as np
import cv2

import std_msgs.msg
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

class PoseFusionNode(Node):
    def __init__(self):
        super().__init__('pose_fusion_node')

        self.publisher_ = self.create_publisher(PointCloud2, 'pose_3d', 10)

        # Subscribers per i rilevamenti da due telecamere - Aggiusta i nomi dei topics come necessario
        self.detections_sub1 = Subscriber(self, DetectionArray, '/camera1/yolo/detections')
        self.detections_sub2 = Subscriber(self, DetectionArray, '/camera2/yolo/detections')
        
        # Sincronizzatore per i rilevamenti
        self.sync = ApproximateTimeSynchronizer([self.detections_sub1, self.detections_sub2], queue_size=10, slop=0.5)
        self.sync.registerCallback(self.detections_callback)

    def detections_callback(self, detections1, detections2):
        # Questa funzione verrà chiamata quando i messaggi da entrambe le telecamere sono sincronizzati
        # Estrai i keypoints da ciascun array di detections e fai la fusione
        keypoints1 = self.extract_keypoints(detections1)
        keypoints2 = self.extract_keypoints(detections2)

        if keypoint1 and keypoint2:
            # Calcola le pose 3D dalla triangolazione o altri metodi
            pose_3d = self.fuse_poses(keypoints1, keypoints2)
            # Pubblica le pose 3D
            self.publish_pose(pose_3d)

    def extract_keypoints(self, detections):
        # Estrae i keypoints da ogni detection
        keypoints_list = []
        for detection in detections.detections:
            for keypoint in detection.keypoints.data:
                keypoints_list.append((keypoint.point.x, keypoint.point.y, keypoint.score))
        return keypoints_list

    def fuse_poses(self, keypoints1, keypoints2):
        # Assumi che self.camera_params contenga le matrici di proiezione P1 e P2 per le telecamere
        P1 = self.camera_params["camera1"]["P"]
        P2 = self.camera_params["camera2"]["P"]

        # Converti le liste di keypoints in array NumPy appropriati
        points1 = np.array([kp[:2] for kp in keypoints1]).T
        points2 = np.array([kp[:2] for kp in keypoints2]).T

        # Homogeneizza i punti (aggiungi una fila di 1)
        points1 = np.vstack((points1, np.ones((1, points1.shape[1]))))
        points2 = np.vstack((points2, np.ones((1, points2.shape[1]))))

        # Applica la triangolazione
        points_4d_hom = cv2.triangulatePoints(P1, P2, points1, points2)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Converti da coordinate omogenee a 3D

        return points_3d.T  # Transponi per avere una lista di coordinate (x, y, z)
        

    def publish_pose(self, pose_3d):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"  # Adatta al tuo frame di riferimento

        keypoints_msg = KeyPoint3DArray()
        keypoints_msg.header = header
        keypoints_msg.data = [KeyPoint3D(x=p[0], y=p[1], z=p[2]) for p in pose_3d]

        self.publisher_.publish(keypoints_msg)
        self.get_logger().info('Published 3D pose')

def main(args=None):
    rclpy.init(args=args)
    node = PoseFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()