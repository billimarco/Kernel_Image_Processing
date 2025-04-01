#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <optional>

#include <sys/stat.h>  // Per creare cartelle
#include <sys/types.h>
#ifdef _WIN32
    #include <direct.h>
    #define MKDIR(path) _mkdir(path)
#else
    #include <unistd.h>
    #define MKDIR(path) mkdir(path, 0777)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;
// change OMP_NUM_THREADS environment variable to run with 1 to X threads...
// check configuration in drop down menu
// XXX check working directory so that ./images and ./output are valid !

struct STBImage {
    int width{0}, height{0}, channels{0};
    uint8_t *image_data{nullptr};
    std::string filename{};

    // Funzione per caricare un'immagine
    bool loadImage(const std::string &name) {
        image_data = stbi_load(name.c_str(), &width, &height, &channels, 3); // Immagine rgb (3 canale) rimuovendo un eventuale canale Alpha
        if (channels == 4)
            channels = 3; //questo perchè se le immagini caricate sono RGBA ritorna il valore 4 su channels anche se quelli caricati sono solo 3
        if (!image_data)
            return false;
        else {
            filename = name;
            return true;
        }
    }

    // Funzione per salvare l'immagine
    void saveImage(const std::string &newName) const {
        stbi_write_jpg(newName.c_str(), width, height, channels, image_data, width);
    }

    // Funzione per inizializzare un'immagine RGB
    void initializeRGB(int w, int h) {
        width = w;
        height = h;
        channels = 3; // Immagine rgb con 3 canale
        image_data = (uint8_t*)malloc(width * height * channels);
    }
};

struct Kernel {
    std::vector<std::vector<float>> matrix;
    int size;

    // Costruttore
    Kernel(int s) : size(s) {
        matrix.resize(size, std::vector<float>(size, 0));
    }

    // Costruttore con inizializzazione e normalizzazione opzionale
    Kernel(int s, std::vector<std::vector<float>> initMatrix, bool normalizeKernel = true) : size(s) {
        if (initMatrix.size() != size || anyRowInvalid(initMatrix)) {
            throw std::invalid_argument("La matrice deve essere quadrata e della dimensione specificata.");
        }
        matrix = initMatrix;
        if (normalizeKernel) {
            normalize();
        }
    }

    // Metodo per verificare che tutte le righe abbiano la lunghezza corretta
    bool anyRowInvalid(const std::vector<std::vector<float>>& mat) {
        for (const auto& row : mat) {
            if (row.size() != size) {
                return true;
            }
        }
        return false;
    }

    // Metodo per normalizzare il kernel
    void normalize() {
        float sum = 0.0;
        for (const auto& row : matrix) {
            for (float value : row) {
                sum += value;
            }
        }
        if (sum != 0) {
            for (auto& row : matrix) {
                for (float& value : row) {
                    value /= sum;
                }
            }
        }
    }

    // Funzione per verificare se il kernel è separabile
    bool isSeparable() const {
        // Convertiamo la matrice in una matrice Eigen
        Eigen::MatrixXf mat(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                mat(i, j) = matrix[i][j];
            }
        }
    
        // Calcoliamo la decomposizione SVD della matrice
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
        // Otteniamo i valori singolari
        Eigen::VectorXf singularValues = svd.singularValues();

        //std::cout << "SingularValues SVD (se uno solo è un valore non nullo è separabile):\n" << singularValues << std::endl;
    
        // Verifica se il numero di valori singolari non nulli è 1 (indica che la matrice ha rango 1)
        return singularValues(0) > 1e-6 && singularValues.tail(singularValues.size() - 1).isZero(1e-6);
    }

    bool separate(std::vector<float>& vertical, std::vector<float>& horizontal) const {
        // Convertiamo la matrice in una matrice Eigen
        Eigen::MatrixXf mat(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                mat(i, j) = matrix[i][j];
            }
        }
    
        // Calcoliamo la decomposizione SVD della matrice
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
        // Otteniamo i valori singolari
        Eigen::VectorXf singularValues = svd.singularValues();

        //std::cout << "SingularValues SVD (se uno solo è un valore non nullo è separabile):\n" << singularValues << std::endl;
    
        // Se il rango è maggiore di 1, il kernel non è separabile
        if (singularValues(0) < 1e-6 || !singularValues.tail(singularValues.size() - 1).isZero(1e-6)) {
            return false;
        }
    
        // Otteniamo i vettori singolari
        Eigen::VectorXf u = svd.matrixU().col(0);  // Primo vettore di U (vettore colonna)
        Eigen::VectorXf v = svd.matrixV().col(0);  // Primo vettore di V (vettore riga)
    
        // Copiamo i vettori in std::vector
        vertical.assign(u.data(), u.data() + u.size());  // Vettore colonna in v
        horizontal.assign(vertical.data(), vertical.data() + vertical.size());  // Vettore riga in h

        //std::cout << "Vettore di convoluzione verticale:\n" << u << std::endl;
        //std::cout << "Vettore di convoluzione orizzontale:\n" << v << std::endl;
    
        return true;
    }

    // Metodo per stampare il kernel
    void print() const {
        for (const auto& row : matrix) {
            for (float value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Funzione per creare un cammino di cartelle
void createPath(const std::string &path) {
    std::istringstream ss(path);
    std::string partialPath;
    std::vector<std::string> directories;
    
    // Dividere il percorso nelle singole directory
    while (std::getline(ss, partialPath, '/')) {
        directories.push_back(partialPath);
    }

    std::string currentPath;
    for (const auto &dir : directories) {
        if (!currentPath.empty()) {
            currentPath += "/";
        }
        currentPath += dir;

        struct stat info;
        if (stat(currentPath.c_str(), &info) != 0) { // Se la cartella non esiste
            MKDIR(currentPath.c_str());
        }
    }
}

// Funzione per caricare immagini in un vettore
std::vector<STBImage> loadImages(const std::string& directory) {
    std::vector<STBImage> images;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().string();
            STBImage img;
            if (img.loadImage(filename)) {
                images.push_back(img);
            }
        }
    }
    return images;
}

STBImage convolveRGB(const STBImage &img, const Kernel &kernel) {
    int kCenter = kernel.size / 2;

    STBImage outputImg;
    outputImg.initializeRGB(img.width, img.height);

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int c = 0; c < img.channels; c++) {
                float sum = 0;
                for (int i = 0; i < kernel.size; i++) {
                    for (int j = 0; j < kernel.size; j++) {
                        int nx = x + j - kCenter;
                        int ny = y + i - kCenter;

                        // Zero-padding: Se fuori dai bordi, usa 0
                        if (nx < 0 || nx >= img.width || ny < 0 || ny >= img.height) {
                            sum += 0;
                        } else {
                            int pixelIndex = (ny * img.width + nx) * img.channels + c;
                            sum += img.image_data[pixelIndex] * kernel.matrix[i][j];
                        }
                    }
                }
                int newIndex = (y * img.width + x) * img.channels + c;
                outputImg.image_data[newIndex] = std::min(std::max(int(sum), 0), 255);
            }
        }
    }
    return outputImg;
}

STBImage separableConvolutionRGB(const STBImage &img, Kernel &kernel) {
    std::vector<float> v, h;
    if (!kernel.separate(v, h)) {
        throw std::runtime_error("Il kernel non è separabile!");
    }

    int kCenter = kernel.size / 2;

    STBImage tempImg, outputImg;
    tempImg.initializeRGB(img.width, img.height);
    outputImg.initializeRGB(img.width, img.height);

    // Passo 1: Convoluzione orizzontale con zero-padding
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int c = 0; c < img.channels; c++) {
                float sum = 0;
                for (int k = 0; k < kernel.size; k++) {
                    int nx = x + k - kCenter;  // Calcola l'indice in base a k

                    // Zero padding: se l'indice è fuori dai limiti, usa zero
                    if (nx < 0 || nx >= img.width) {
                        sum += 0;  // Zero padding
                    } else {
                        int index = (y * img.width + nx) * img.channels + c;
                        sum += img.image_data[index] * h[k];
                    }
                }
                int tempIndex = (y * img.width + x) * img.channels + c;
                tempImg.image_data[tempIndex] = std::min(std::max(int(sum), 0), 255);
            }
        }
    }

    // Passo 2: Convoluzione verticale con zero-padding
    for (int y = 0; y < tempImg.height; y++) {
        for (int x = 0; x < tempImg.width; x++) {
            for (int c = 0; c < tempImg.channels; c++) {
                float sum = 0;
                for (int k = 0; k < kernel.size; k++) {
                    int ny = y + k - kCenter;  // Calcola l'indice in base a k

                    // Zero padding: se l'indice è fuori dai limiti, usa zero
                    if (ny < 0 || ny >= tempImg.height) {
                        sum += 0;  // Zero padding
                    } else {
                        int index = (ny * tempImg.width + x) * tempImg.channels + c;
                        sum += tempImg.image_data[index] * v[k];
                    }
                }
                int finalIndex = (y * tempImg.width + x) * tempImg.channels + c;
                outputImg.image_data[finalIndex] = std::min(std::max(int(sum), 0), 255);
            }
        }
    }

    delete[] tempImg.image_data;  // Libera la memoria temporanea
    return outputImg;
}

int main(){
    createPath("images/convolution");
    createPath("images/separable_convolution");
    std::ifstream conf_file("settings/config.json");
    json config = json::parse(conf_file);

    int num_images = config["num_images"];

    std::vector<STBImage> loadedImages = loadImages("images/basis");
    std::cout << "Totale immagini caricate: " << loadedImages.size() << std::endl;

    try {
        std::vector<std::vector<float>> sharpening = {{0, -1, 0}, 
                                                      {-1, 5, -1}, 
                                                      {0, -1, 0}};  // Esempio di Sharpen
        
        std::vector<std::vector<float>> gaussian_blur = {{1, 2, 1}, 
                                                         {2, 4, 2}, 
                                                         {1, 2, 1}}; // Esempio di Blur Gaussiano
        Kernel k(3, gaussian_blur, true);
        k.print();

        // Applica la convoluzione a tutte le immagini
        for (size_t i = 0; i < loadedImages.size(); i++) {
            STBImage result1 = convolveRGB(loadedImages[i], k);
            std::string originalFilename1 = std::filesystem::path(loadedImages[i].filename).filename().string();
            std::string outputFilename1 = "images/convolution/" + originalFilename1;
            result1.saveImage(outputFilename1);
            std::cout << "Immagine salvata come " << outputFilename1 << std::endl;

            if(k.isSeparable()){
                STBImage result2 = separableConvolutionRGB(loadedImages[i], k);
                std::string originalFilename2 = std::filesystem::path(loadedImages[i].filename).filename().string();
                std::string outputFilename2 = "images/separable_convolution/" + originalFilename2;
                result1.saveImage(outputFilename2);
                std::cout << "Immagine salvata come " << outputFilename2 << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }

    return 0;
}
