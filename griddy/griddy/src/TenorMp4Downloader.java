import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.io.IOException;
import org.json.JSONObject;
import org.json.JSONArray;



public class TenorMp4Downloader {

    private static final String API_KEY = "AIzaSyDHzx2dw4f0ED716XI8hzwm3W44exqhzHk";  // Replace with your Tenor API key
    private static final String SEARCH_TERM = "floss";
    private static final int LIMIT = 100;

    public static void main(String[] args) {
        try {
            // Create HttpClient and make request
            HttpClient client = HttpClient.newHttpClient();
            String url = "https://tenor.googleapis.com/v2/search?q=" + SEARCH_TERM + "&key=" + API_KEY + "&limit=" + LIMIT;
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .build();

            // Get response and handle it
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) {
                JSONObject jsonResponse = new JSONObject(response.body());
                JSONArray results = jsonResponse.getJSONArray("results");

                for (int i = 0; i < results.length(); i++) {
                    JSONObject result = results.getJSONObject(i);
                    String mp4Url = result.getJSONObject("media_formats").getJSONObject("mp4").getString("url");
                    System.out.println("MP4 URL: " + mp4Url);

                    // Download the MP4 file using HttpClient
                    downloadMp4(mp4Url, SEARCH_TERM + i + ".mp4");
                }
            } else {
                System.out.println("Failed to fetch MP4s, status code: " + response.statusCode());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Method to download MP4 using HttpClient
    private static void downloadMp4(String mp4Url, String outputFilePath) throws IOException, InterruptedException {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(mp4Url))  // Using URI.create to avoid URL constructor
                .build();

        // Create the path for the file
        Path path = Paths.get(outputFilePath);

        // Send the request and download the file as a byte stream
        HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());

        // Write the downloaded bytes to a file
        Files.write(path, response.body(), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
        
        System.out.println("Saved MP4: " + outputFilePath);

        //System.out.println("retfghugfdrijuytrghjkhgftdrgvhbjgytfghvbnjhgy");
    }
}
