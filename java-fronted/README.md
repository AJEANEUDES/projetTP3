# Interface Java pour la Reconnaissance de Dessins

## Prérequis
- Java 17 ou supérieur
- Maven
- JavaFX (inclus dans les dépendances)

## Configuration

1. Compiler le projet :
```bash
mvn clean install
```

2. Lancer l'application :
```bash
mvn javafx:run
```

## Structure du Projet

- `SketchRecognitionApplication.java` : Point d'entrée de l'application
- `controller/` : Contrôleurs JavaFX
- `resources/fxml/` : Fichiers de layout FXML
- `resources/application.properties` : Configuration Spring Boot

/controller/MainController.java