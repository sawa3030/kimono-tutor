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
  const [pictureURI, setPictureURI] = useState<null | string>(null);

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
      setPictureURI(photo.uri);
    }
  };

  function dataURItoBlob(dataURI: string) {
    var byteString;
    if (dataURI.split(",")[0].indexOf("base64") >= 0)
      byteString = atob(dataURI.split(",")[1]);
    else byteString = unescape(dataURI.split(",")[1]);

    var mimeString = dataURI.split(",")[0].split(":")[1].split(";")[0];

    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], { type: mimeString });
  }

  const postPicture = async () => {
    if (pictureURI) {
      const formData = new FormData();

      var blob = dataURItoBlob(pictureURI);
      formData.append("picture", blob);

      const uploadResponse = await fetch("http://localhost:8000", {
        method: "POST",
        body: formData,
      });
      const result = await uploadResponse.json();
      console.log(result);
    }
  };

  return (
    <View style={styles.container}>
      {pictureURI ? (
        <>
          <TouchableOpacity onPress={postPicture}>
            <Text>Post Picture</Text>
          </TouchableOpacity>
          <Image source={{ uri: pictureURI }} style={styles.camera} />
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
