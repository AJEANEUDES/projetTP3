package com.sketchrecognition.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class RecognitionService {
    
    @Value("${recognition.api.url}")
    private String apiUrl;
    
    private final RestTemplate restTemplate;
    
    public RecognitionService() {
        this.restTemplate = new RestTemplate();
    }
    
    public Map<String, Object> recognizeDrawing(double[][] imageData) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("image", imageData);
            
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);
            
            ResponseEntity<Map> response = restTemplate.postForEntity(
                apiUrl + "/predict",
                request,
                Map.class
            );
            
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return response.getBody();
            } else {
                throw new RuntimeException("Failed to get prediction from API");
            }
        } catch (Exception e) {
            throw new RuntimeException("Error calling recognition API: " + e.getMessage(), e);
        }
    }
    
    public boolean checkHealth() {
        try {
            ResponseEntity<Map> response = restTemplate.getForEntity(
                apiUrl + "/health",
                Map.class
            );
            return response.getStatusCode().is2xxSuccessful() &&
                   response.getBody() != null &&
                   "healthy".equals(response.getBody().get("status"));
        } catch (Exception e) {
            return false;
        }
    }
}