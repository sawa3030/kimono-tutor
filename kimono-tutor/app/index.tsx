import { Link } from "expo-router";
import { Text, View } from "react-native";
import { useRouter } from 'expo-router';
import { Button } from 'react-native';

export default function Index() {
  const router = useRouter();

  const handlePress = () => {
    router.push('/kitsuke');
  };

  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Text>Kimono-Tutorへようこそ!!!</Text>
      <Button title="着付け教室へ" onPress={handlePress} />
    </View>
  );
}
