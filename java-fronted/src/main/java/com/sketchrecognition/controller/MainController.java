package com.sketchrecognition.controller;

import com.sketchrecognition.service.RecognitionService;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Alert;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;

@Controller
public class MainController {

    @FXML
    private Canvas canvas;
    
    @FXML
    private Label resultLabel;
    
    @FXML
    private Label confidenceLabel;
    
    @FXML
    private VBox predictionsList;
    
    @FXML
    private ProgressIndicator progressIndicator;

    private GraphicsContext gc;
    private double lastX, lastY;
    
    @Autowired
    private RecognitionService recognitionService;

    @FXML
    public void initialize() {
        gc = canvas.getGraphicsContext2D();
        setupCanvas();
        checkApiHealth();
    }

    private void setupCanvas() {
        gc.setLineWidth(3.0);
        gc.setStroke(Color.BLACK);
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        
        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, e -> {
            lastX = e.getX();
            lastY = e.getY();
            gc.beginPath();
            gc.moveTo(lastX, lastY);
            gc.stroke();
        });

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, e -> {
            gc.lineTo(e.getX(), e.getY());
            gc.stroke();
            lastX = e.getX();
            lastY = e.getY();
        });
        
        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED, e -> {
            gc.closePath();
        });
    }

    private void checkApiHealth() {
        new Thread(() -> {
            boolean isHealthy = recognitionService.checkHealth();
            Platform.runLater(() -> {
                if (!isHealthy) {
                    showError("API Error", "Cannot connect to recognition API. Please ensure the backend server is running.");
                }
            });
        }).start();
    }

    @FXML
    private void clearCanvas() {
        gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        resultLabel.setText("");
        confidenceLabel.setText("");
        predictionsList.getChildren().clear();
    }

    @FXML
    private void recognizeDrawing() {
        progressIndicator.setVisible(true);
        resultLabel.setText("Reconnaissance en cours...");
        confidenceLabel.setText("");
        predictionsList.getChildren().clear();
        
        WritableImage writableImage = new WritableImage((int) canvas.getWidth(), (int) canvas.getHeight());
        canvas.snapshot(null, writableImage);
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(writableImage, null);
        
        double[][] imageData = convertToPixelArray(bufferedImage);
        
        new Thread(() -> {
            try {
                Map<String, Object> result = recognitionService.recognizeDrawing(imageData);
                
                Platform.runLater(() -> {
                    progressIndicator.setVisible(false);
                    
                    String className = (String) result.get("class");
                    double confidence = (double) result.get("confidence");
                    
                    resultLabel.setText("Prédiction : " + className);
                    confidenceLabel.setText(String.format("Confiance : %.1f%%", confidence * 100));
                    
                    // Affichage des top prédictions
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> topPredictions = (List<Map<String, Object>>) result.get("top_predictions");
                    if (topPredictions != null) {
                        predictionsList.getChildren().clear();
                        for (Map<String, Object> pred : topPredictions) {
                            String predClass = (String) pred.get("class");
                            double predConf = (double) pred.get("confidence");
                            Label predLabel = new Label(String.format("%s (%.1f%%)", predClass, predConf * 100));
                            predictionsList.getChildren().add(predLabel);
                        }
                    }
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    progressIndicator.setVisible(false);
                    showError("Erreur", "Erreur lors de la reconnaissance : " + e.getMessage());
                });
            }
        }).start();
    }
    
    private double[][] convertToPixelArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] result = new double[height][width];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int gray = (int) (((rgb >> 16) & 0xFF) * 0.299 + 
                                ((rgb >> 8) & 0xFF) * 0.587 + 
                                (rgb & 0xFF) * 0.114);
                result[y][x] = gray / 255.0;
            }
        }
        
        return result;
    }
    
    private void showError(String title, String content) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }
}