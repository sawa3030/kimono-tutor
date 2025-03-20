import { useState } from "react";
import {
  Button,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Image,
} from "react-native";
import {
  CameraView,
  CameraType,
  useCameraPermissions,
  Camera,
} from "expo-camera";

export default function Index() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraRef, setCamera] = useState<CameraView | null>(null);
  const [picture, setPicture] = useState<null | string>(null);

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View>
        <Text>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  const takePicture = async () => {
    if (cameraRef) {
      const photo = await cameraRef.takePictureAsync();
      console.log(photo);
      console.log("took photo");
      if (!photo) {
        console.log("no photo");
        return;
      }
      setPicture(photo.uri);
    }
  };

  const postPicture = async () => {
    if (picture) {
      const formData = new FormData();
      // formData.append("picture", picture);
      // formData.append("number", "1");
      const data = { number: "1" };

      const uploadResponse = await fetch("http://127.0.0.1:8000", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // body: formData,
        body: JSON.stringify(data),
      });
      const result = await uploadResponse.json();
      console.log(result);
    }
  };

  return (
    <View style={styles.container}>
      {picture ? (
        <>
          <TouchableOpacity onPress={postPicture}>
            <Text>Post Picture</Text>
          </TouchableOpacity>
          <Image source={{ uri: picture }} style={styles.camera} />
        </>
      ) : (
        <CameraView
          style={styles.camera}
          facing={facing}
          ref={(ref) => {
            setCamera(ref);
          }}
        >
          <View>
            <TouchableOpacity onPress={toggleCameraFacing}>
              <Text>Flip Camera</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={takePicture}>
              <Text>Take Picture</Text>
            </TouchableOpacity>
          </View>
        </CameraView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  camera: {
    flex: 1,
  },
});
