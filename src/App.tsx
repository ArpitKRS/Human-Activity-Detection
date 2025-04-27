import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { Camera, ActivitySquare } from 'lucide-react';

// Activities we can detect based on pose
const activities = ['Standing', 'Movement', 'Raising Hands', 'Stable', 'Waving', 'Sitting'];

function App() {
  const webcamRef = useRef<Webcam>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(
    null
  );
  const [currentActivity, setCurrentActivity] = useState<string>('Stable');
  const [isLoading, setIsLoading] = useState(true);
  const [previousPositions, setPreviousPositions] = useState<
    Array<{ x: number; y: number }>
  >([]);
  const [isWebcamReady, setIsWebcamReady] = useState(false);
  const [hasVideoStream, setHasVideoStream] = useState(false);
  // Track wrist positions for waving detection
  const [wristPositions, setWristPositions] = useState<
    Array<{ left: { x: number; y: number }; right: { x: number; y: number } }>
  >([]);

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const model = poseDetection.SupportedModels.MoveNet;
      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
        minPoseScore: 0.3,
      };
      const detector = await poseDetection.createDetector(
        model,
        detectorConfig
      );
      setDetector(detector);
      setIsLoading(false);
    };

    loadModel();
  }, []);

  // Add new effect to monitor video stream status
  useEffect(() => {
    const checkVideoStream = () => {
      const video = webcamRef.current?.video;
      if (
        video &&
        video.readyState === 4 &&
        video.videoWidth > 0 &&
        video.videoHeight > 0
      ) {
        setHasVideoStream(true);
      } else {
        setHasVideoStream(false);
      }
    };

    const interval = setInterval(checkVideoStream, 100);
    return () => clearInterval(interval);
  }, []);

  const calculateAngle = (
    p1: poseDetection.Keypoint,
    p2: poseDetection.Keypoint,
    p3: poseDetection.Keypoint
  ) => {
    const radians =
      Math.atan2(p3.y - p2.y, p3.x - p2.x) -
      Math.atan2(p1.y - p2.y, p1.x - p2.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
  };

  const calculateDistance = (
    p1: poseDetection.Keypoint,
    p2: poseDetection.Keypoint
  ) => {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
  };

  const isMoving = (currentPos: { x: number; y: number }) => {
    if (previousPositions.length < 5) return false;

    const recentPositions = previousPositions.slice(-5);
    const avgMovement =
      recentPositions.reduce((acc, pos, idx, arr) => {
        if (idx === 0) return 0;
        const prevPos = arr[idx - 1];
        return (
          acc +
          Math.sqrt(
            Math.pow(pos.x - prevPos.x, 2) + Math.pow(pos.y - prevPos.y, 2)
          )
        );
      }, 0) /
      (recentPositions.length - 1);

    return avgMovement > 15; // Threshold for movement detection
  };

  // Function to detect waving motion
  const isWaving = (wristHistory: Array<{ left: { x: number; y: number }; right: { x: number; y: number } }>) => {
    if (wristHistory.length < 10) return false; // Need enough history to detect wave pattern
    
    // Looking at just the last 10 frames for waving detection
    const recentWristPositions = wristHistory.slice(-10);
    
    // Check for horizontal movement in either wrist
    const leftWristXChanges = recentWristPositions.map((pos, idx, arr) => {
      if (idx === 0) return 0;
      return Math.abs(pos.left.x - arr[idx - 1].left.x);
    }).slice(1);
    
    const rightWristXChanges = recentWristPositions.map((pos, idx, arr) => {
      if (idx === 0) return 0;
      return Math.abs(pos.right.x - arr[idx - 1].right.x);
    }).slice(1);
    
    // Calculate average horizontal movement
    const avgLeftWristXChange = leftWristXChanges.reduce((sum, val) => sum + val, 0) / leftWristXChanges.length;
    const avgRightWristXChange = rightWristXChanges.reduce((sum, val) => sum + val, 0) / rightWristXChanges.length;
    
    // Check for vertical stability (waving should be mostly horizontal)
    const leftWristYChanges = recentWristPositions.map((pos, idx, arr) => {
      if (idx === 0) return 0;
      return Math.abs(pos.left.y - arr[idx - 1].left.y);
    }).slice(1);
    
    const rightWristYChanges = recentWristPositions.map((pos, idx, arr) => {
      if (idx === 0) return 0;
      return Math.abs(pos.right.y - arr[idx - 1].right.y);
    }).slice(1);
    
    const avgLeftWristYChange = leftWristYChanges.reduce((sum, val) => sum + val, 0) / leftWristYChanges.length;
    const avgRightWristYChange = rightWristYChanges.reduce((sum, val) => sum + val, 0) / rightWristYChanges.length;
    
    // Waving is detected when:
    // 1. Significant horizontal movement in either wrist
    // 2. Horizontal movement is significantly greater than vertical movement
    // 3. Movement has some consistency (not just random)
    
    const leftWristWaving = avgLeftWristXChange > 20 && avgLeftWristXChange > 2 * avgLeftWristYChange;
    const rightWristWaving = avgRightWristXChange > 20 && avgRightWristXChange > 2 * avgRightWristYChange;
    
    return leftWristWaving || rightWristWaving;
  };

  const detectPose = async () => {
    if (!detector || !webcamRef.current) return;

    const video = webcamRef.current.video;
    if (!video || !isWebcamReady || !hasVideoStream) return;

    // Additional checks for video readiness
    if (
      video.readyState !== 4 ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    )
      return;

    try {
      const pose = await detector.estimatePoses(video);

      if (pose.length > 0 && pose[0].score && pose[0].score > 0.3) {
        const keypoints = pose[0].keypoints;
        const activity = determineActivity(keypoints);
        setCurrentActivity(activity);

        // Update position history for movement detection
        const nose = keypoints.find((kp) => kp.name === 'nose');
        if (nose) {
          setPreviousPositions((prev) => [
            ...prev.slice(-9),
            { x: nose.x, y: nose.y },
          ]);
        }

        // Update wrist position history for waving detection
        const leftWrist = keypoints.find((kp) => kp.name === 'left_wrist');
        const rightWrist = keypoints.find((kp) => kp.name === 'right_wrist');
        if (leftWrist && rightWrist) {
          setWristPositions((prev) => [
            ...prev.slice(-19),
            {
              left: { x: leftWrist.x, y: leftWrist.y },
              right: { x: rightWrist.x, y: rightWrist.y }
            }
          ]);
        }
      }
    } catch (error) {
      console.error('Error during pose detection:', error);
      // Reset states if we encounter an error
      setHasVideoStream(false);
    }
  };

  const determineActivity = (keypoints: poseDetection.Keypoint[]) => {
    if (!keypoints || keypoints.length === 0) return 'Stable';

    const findKeypoint = (name: string) =>
      keypoints.find((kp) => kp.name === name);

    const nose = findKeypoint('nose');
    const leftShoulder = findKeypoint('left_shoulder');
    const rightShoulder = findKeypoint('right_shoulder');
    const leftElbow = findKeypoint('left_elbow');
    const rightElbow = findKeypoint('right_elbow');
    const leftWrist = findKeypoint('left_wrist');
    const rightWrist = findKeypoint('right_wrist');
    const leftHip = findKeypoint('left_hip');
    const rightHip = findKeypoint('right_hip');
    const leftKnee = findKeypoint('left_knee');
    const rightKnee = findKeypoint('right_knee');
    const leftAnkle = findKeypoint('left_ankle');
    const rightAnkle = findKeypoint('right_ankle');

    if (
      !nose ||
      !leftShoulder ||
      !rightShoulder ||
      !leftElbow ||
      !rightElbow ||
      !leftWrist ||
      !rightWrist ||
      !leftHip ||
      !rightHip ||
      !leftKnee ||
      !rightKnee ||
      !leftAnkle ||
      !rightAnkle
    ) {
      return 'Stable';
    }

    // Calculate key angles and distances
    const leftKneeAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
    const rightKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
    const leftHipAngle = calculateAngle(leftShoulder, leftHip, leftKnee);
    const rightHipAngle = calculateAngle(rightShoulder, rightHip, rightKnee);

    // Vertical distances and ratios
    const shoulderHipDist =
      (calculateDistance(leftShoulder, leftHip) +
        calculateDistance(rightShoulder, rightHip)) /
      2;
    const hipKneeDist =
      (calculateDistance(leftHip, leftKnee) +
        calculateDistance(rightHip, rightKnee)) /
      2;
    const kneeAnkleDist =
      (calculateDistance(leftKnee, leftAnkle) +
        calculateDistance(rightKnee, rightAnkle)) /
      2;

    // Height ratios (useful for posture detection)
    const totalHeight = shoulderHipDist + hipKneeDist + kneeAnkleDist;
    const shoulderHipRatio = shoulderHipDist / totalHeight;
    const hipKneeRatio = hipKneeDist / totalHeight;

    // Check if arms are raised (both arms above shoulders)
    if (
      leftWrist.y < leftShoulder.y - 50 &&
      rightWrist.y < rightShoulder.y - 50
    ) {
      return 'Raising Hands';
    }

    // Check for waving (horizontal movement of wrists)
    if (isWaving(wristPositions)) {
      return 'Waving';
    }

    // Check for squatting
    // Characteristics: bent knees, lowered hips, upright torso
    const isSquatting =
      leftKneeAngle < 100 &&
      rightKneeAngle < 100 && // Bent knees
      leftHipAngle > 45 &&
      rightHipAngle > 45 && // Hip flexion
      Math.abs(leftShoulder.y - rightShoulder.y) < 30 && // Level shoulders (upright torso)
      nose.y > (leftShoulder.y + rightShoulder.y) / 2; // Head above shoulders

    if (isSquatting) {
      return 'Squatting';
    }

    // Check for sitting
    // Characteristics: knees bent around 90 degrees, hips lower than knees
    const isSitting =
      leftKneeAngle < 120 &&
      rightKneeAngle < 120 && // Bent knees
      leftHip.y > leftKnee.y - 20 && // Hips approximately at knee level or slightly above
      rightHip.y > rightKnee.y - 20 &&
      shoulderHipRatio < 0.4 && // Compressed vertical distance between shoulders and hips
      hipKneeRatio > 0.3; // Significant distance between hips and knees

    if (isSitting) {
      return 'Sitting';
    }

    // Checking for movement
    if (isMoving({ x: nose.x, y: nose.y })) {
      return 'Movement';
    }

    // Check for standing
    // Characteristics: straight legs, vertical alignment, even weight distribution
    const isStanding =
      leftKneeAngle > 160 &&
      rightKneeAngle > 160 && // Straight legs
      Math.abs(leftShoulder.y - rightShoulder.y) < 30 && // Level shoulders
      Math.abs(leftHip.y - rightHip.y) < 30 && // Level hips
      Math.abs(leftKnee.y - rightKnee.y) < 30 && // Level knees
      shoulderHipRatio > 0.3 && // Normal vertical spacing
      hipKneeRatio > 0.3; // Normal vertical spacing

    if (isStanding) {
      return 'Standing';
    }

    return 'Stable';
  };

  useEffect(() => {
    const interval = setInterval(() => {
      detectPose();
    }, 100);

    return () => clearInterval(interval);
  }, [detector, previousPositions, wristPositions, isWebcamReady, hasVideoStream]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col items-center space-y-8">
          <div className="flex items-center space-x-3">
            <Camera className="w-8 h-8" />
            <h1 className="text-3xl font-bold">Human Activity Detection</h1>
            <ActivitySquare className="w-8 h-8" />
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center h-[480px] w-[640px] bg-gray-800 rounded-lg">
              <div className="text-xl">Loading Model...</div>
            </div>
          ) : (
            <div className="relative">
              <Webcam
                ref={webcamRef}
                className="rounded-lg shadow-xl"
                width={640}
                height={480}
                mirrored={true}
                onUserMedia={() => setIsWebcamReady(true)}
              />
              <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur-sm px-4 py-2 rounded-full">
                <p className="text-lg">
                  Activity:{' '}
                  <span className="font-semibold text-green-400">
                    {currentActivity}
                  </span>
                </p>
              </div>
            </div>
          )}

          <div className="bg-gray-800 p-6 rounded-lg w-full max-w-2xl">
            <h2 className="text-xl font-semibold mb-4">
              Detectable Activities:
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {activities.map((activity) => (
                <div
                  key={activity}
                  className={`p-3 rounded-lg text-center ${
                    currentActivity === activity
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700'
                  }`}
                >
                  {activity}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;