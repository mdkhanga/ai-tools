package com.mj.refund.controller;

import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@RestController
@RequestMapping("/v1/refunds")
public class RefundController {
    private final RestTemplate restTemplate = new RestTemplate();
    private static final String MODEL_SERVER_URL = "http://127.0.0.1:8000/predict";
    private final Random random = new Random();

    @GetMapping("/latest")
    public ResponseEntity<?> getLatestRefundPrediction() {

        // Generate random input values
        int income = 50000 + random.nextInt(200001); // 50,000–250,000
        int dependents = 2 + random.nextInt(3);      // 2–4
        int withholding = 5000 + random.nextInt(20001); // 5,000–25,000
        int filingStatus = random.nextInt(2);        // 0 or 1


        // Build request body with some random values
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("income", income);
        requestBody.put("dependents", dependents);
        requestBody.put("withholding", withholding);
        requestBody.put("filing_status", filingStatus);

        // Set headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Build HTTP entity
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

        // Call the Python model server
        ResponseEntity<String> response =
                restTemplate.postForEntity(MODEL_SERVER_URL, entity, String.class);

        // Return the response directly
        return ResponseEntity.ok(response.getBody());
    }


}
