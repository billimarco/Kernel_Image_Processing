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
        image_data = stbi_load(name.c_str(), &width, &height, &channels, 3); // Immagine binaria (1 canale)
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
    bool isSeparable() {
        std::vector<float> firstRow = matrix[0];
        std::vector<float> firstCol(size);

        for (int i = 0; i < size; ++i) {
            firstCol[i] = matrix[i][0];
        }

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (std::abs(matrix[i][j] - firstRow[j] * firstCol[i]) > 1e-6) {
                    return false;
                }
            }
        }
        return true;
    }

    bool separate(std::vector<float>& v, std::vector<float>& h) {
        v = matrix[0];
        h.resize(size, 1.0f);

        for (int i = 0; i < size; ++i) {
            if (matrix[i][0] != 0) {
                float scale = matrix[i][0];
                for (int j = 0; j < size; ++j) {
                    if (std::abs(matrix[i][j] - v[j] * scale) > 1e-6) {
                        return false;
                    }
                }
                h[i] = scale;
            } else {
                return false;
            }
        }
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

STBImage convolveRGB(const STBImage &image, const Kernel &kernel) {
    int height = image.height;
    int width = image.width;
    int kCenter = kernel.size / 2;
    
    STBImage outputImage;
    outputImage.width = width;
    outputImage.height = height;
    outputImage.channels = 3;
    outputImage.image_data = new uint8_t[width * height * 3];
    
    for (int i = kCenter; i < height - kCenter; i++) {
        for (int j = kCenter; j < width - kCenter; j++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0;
                for (int m = 0; m < kernel.size; m++) {
                    for (int n = 0; n < kernel.size; n++) {
                        int pixelIndex = ((i + m - kCenter) * width + (j + n - kCenter)) * 3 + c;
                        sum += image.image_data[pixelIndex] * kernel.matrix[m][n];
                    }
                }
                int newIndex = (i * width + j) * 3 + c;
                outputImage.image_data[newIndex] = std::min(std::max(int(sum), 0), 255);
            }
        }
    }
    return outputImage;
}

int main(){
    createPath("images/modified");
    std::ifstream conf_file("settings/config.json");
    json config = json::parse(conf_file);

    int num_images = config["num_images"];

    std::vector<STBImage> loadedImages = loadImages("images/basis");
    std::cout << "Totale immagini caricate: " << loadedImages.size() << std::endl;
    std::cout << "Totale immagini caricate: " << loadedImages[0].width << std::endl;
    std::cout << "Totale immagini caricate: " << loadedImages[0].height << std::endl;
    std::cout << "Totale immagini caricate: " << loadedImages[0].channels << std::endl;

    try {
        std::vector<std::vector<float>> initMatrix = {{0, -1, 0}, 
                                                      {-1, 5, -1}, 
                                                      {0, -1, 0}};  // Esempio di kernel Gaussiano
        Kernel k(3, initMatrix, false);
        k.print();

        // Applica la convoluzione a tutte le immagini
        for (size_t i = 0; i < loadedImages.size(); ++i) {
            STBImage result = convolveRGB(loadedImages[i], k);
            
            std::string originalFilename = std::filesystem::path(loadedImages[i].filename).filename().string();
            
            std::string outputFilename = "images/modified/" + originalFilename;
            result.saveImage(outputFilename);
            std::cout << "Immagine salvata come " << outputFilename << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }

    return 0;
}
