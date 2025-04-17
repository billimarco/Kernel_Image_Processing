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
#include <chrono>
#include <cuda_runtime.h>

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
const json CONFIG = json::parse(std::ifstream("settings/config.json"));
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
        return singularValues(0) > 1e-6f && singularValues.tail(singularValues.size() - 1).isZero(1e-6f);
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
        if (singularValues(0) < 1e-6f || !singularValues.tail(singularValues.size() - 1).isZero(1e-6f)) {
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

// Dummy CUDA kernel for warmup
__global__ void dummyKernel() {
}

void cudaWarmup() {
    // Launch a small kernel with 1 block and 1 thread (just to initialize GPU)
    auto start = std::chrono::high_resolution_clock::now();
    dummyKernel<<<1, 1>>>();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // Check for any errors that might have occurred
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA warmup fallito: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "CUDA warmup successo:" << std::endl;
        std::cout << "Tempo di esecuzione CUDA warmup: " << duration << " ms" << std::endl;
    }
}
//<------------------------------------------------------------------------>
// Funzioni per la normale convoluzione RGB

// Funzione per fare Image Processing tramite convoluzione di un Kernel
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

// Funzione di convoluzione con un kernel in CUDA
__global__ void convolveKernelRGB(uint8_t *input, uint8_t *output, const float *kernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenter = kernelSize / 2;
    int pixelIndex = (y * width + x) * 3;
    float sum[3] = {0};

    if (x < width && y < height) return;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            int nx = x + j - kCenter;
            int ny = y + i - kCenter;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = (ny * width + nx) * 3;
                float k = kernel[i * kernelSize + j];
                sum[0] += input[idx] * k;
                sum[1] += input[idx + 1] * k;
                sum[2] += input[idx + 2] * k;
            }
        }
    }

    output[pixelIndex] = min(max(int(sum[0]), 0), 255);
    output[pixelIndex + 1] = min(max(int(sum[1]), 0), 255);
    output[pixelIndex + 2] = min(max(int(sum[2]), 0), 255);
}

// Funzione per fare Image Processing tramite convoluzione di un Kernel con CUDA
STBImage convolveRGB_CUDA(const STBImage &img, const Kernel &kernel) {
    std::vector<float> h_allKernel(kernel.size * kernel.size);
    for (int i = 0; i < kernel.size; i++)
        for (int j = 0; j < kernel.size; j++)
            h_allKernel[i * kernel.size + j] = kernel.matrix[i][j];
    
    STBImage outputImg;
    outputImg.initializeRGB(img.width, img.height);
    
    int imageSize = img.width * img.height * 3;
    
    uint8_t *d_input, *d_output;
    float *d_kernel;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernel.size * kernel.size * sizeof(float));
    
    cudaMemcpy(d_input, img.image_data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_allKernel.data(), kernel.size * kernel.size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((img.width + blockSize.x - 1) / blockSize.x, (img.height + blockSize.y - 1) / blockSize.y);
    
    convolveKernelRGB<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, img.width, img.height, kernel.size);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImg.image_data, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return outputImg;
}

//<------------------------------------------------------------------------>
// Funzioni per la convoluzione RGB con kernel separato

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

// CUDA Kernel for the horizontal convolution (first pass)
__global__ void convolutionHorizontalKernelRGB(const unsigned char *input, unsigned char *tempOutput, const float *hKernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenter = kernelSize / 2;
    int pixelIndex = (y * width + x) * 3;
    float sum[3] = {0};

    if (x < width && y < height) return;

    for (int k = 0; k < kernelSize; k++) {
        int nx = x + k - kCenter;
        if (nx >= 0 && nx < width) {
            int idx = (y * width + nx) * 3;

            sum[0] += input[idx] * hKernel[k];
            sum[1] += input[idx + 1] * hKernel[k];
            sum[2] += input[idx + 2] * hKernel[k];
        }
    }

    tempOutput[pixelIndex] = min(max(int(sum[0]), 0), 255);
    tempOutput[pixelIndex + 1] = min(max(int(sum[1]), 0), 255);
    tempOutput[pixelIndex + 2] = min(max(int(sum[2]), 0), 255);    
}

// CUDA Kernel for the vertical convolution (second pass)
__global__ void convolutionVerticalKernelRGB(const unsigned char *tempInput, unsigned char *output, const float *vKernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenter = kernelSize / 2;
    int pixelIndex = (y * width + x) * 3;
    float sum[3] = {0};

    if (x < width && y < height) return;

    
    for (int k = 0; k < kernelSize; k++) {
        int ny = y + k - kCenter;
        if (ny >= 0 && ny < height) {
            int idx = (ny * width + x) * 3;

            sum[0] += tempInput[idx] * vKernel[k];
            sum[1] += tempInput[idx + 1] * vKernel[k];
            sum[2] += tempInput[idx + 2] * vKernel[k];
        }
    }

    output[pixelIndex] = min(max(int(sum[0]), 0), 255);
    output[pixelIndex + 1] = min(max(int(sum[1]), 0), 255);
    output[pixelIndex + 2] = min(max(int(sum[2]), 0), 255); 
}

// Main CUDA function for separable convolution
STBImage separableConvolutionRGB_CUDA(const STBImage &img, Kernel &kernel) {
    std::vector<float> v, h;
    if (!kernel.separate(v, h)) {
        throw std::runtime_error("Il kernel non è separabile!");
    }

    STBImage tempImg, outputImg;
    tempImg.initializeRGB(img.width, img.height);
    outputImg.initializeRGB(img.width, img.height);

    int imageSize = img.width * img.height * 3;

    uint8_t *d_input, *d_tempOutput, *d_output;
    float *d_hKernel, *d_vKernel;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_tempOutput, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_hKernel, kernel.size * sizeof(float));
    cudaMalloc(&d_vKernel, kernel.size * sizeof(float));

    cudaMemcpy(d_input, img.image_data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hKernel, h.data(), kernel.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vKernel, v.data(), kernel.size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((img.width + blockSize.x - 1) / blockSize.x, (img.height + blockSize.y - 1) / blockSize.y);

    convolutionHorizontalKernelRGB<<<gridSize, blockSize>>>(d_input, d_tempOutput, d_hKernel, img.width, img.height, kernel.size);
    cudaDeviceSynchronize();

    convolutionVerticalKernelRGB<<<gridSize, blockSize>>>(d_tempOutput, d_output, d_vKernel, img.width, img.height, kernel.size);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImg.image_data, d_output, img.width * img.height * img.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_tempOutput);
    cudaFree(d_output);
    cudaFree(d_hKernel);
    cudaFree(d_vKernel);

    return outputImg;
}
//<------------------------------------------------------------------------>
int main(){

    createPath("images/convolution");
    createPath("images/convolutionCUDA");
    createPath("images/separable_convolution");
    createPath("images/separable_convolutionCUDA");

    cudaWarmup();

    int num_images = CONFIG["num_images"];

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

            auto start = std::chrono::high_resolution_clock::now();
            STBImage result1 = convolveRGB(loadedImages[i], k);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start).count();
            std::string originalFilename1 = std::filesystem::path(loadedImages[i].filename).filename().string();
            std::string outputFilename1 = "images/convolution/" + originalFilename1;
            result1.saveImage(outputFilename1);
            std::cout << "Immagine salvata come " << outputFilename1 << std::endl;
            std::cout << "Tempo di esecuzione: " << duration << " ms" << std::endl;

            auto start2 = std::chrono::high_resolution_clock::now();
            STBImage result2 = convolveRGB_CUDA(loadedImages[i], k);
            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
            std::string originalFilename2 = std::filesystem::path(loadedImages[i].filename).filename().string();
            std::string outputFilename2 = "images/convolutionCUDA/" + originalFilename2;
            result1.saveImage(outputFilename2);
            std::cout << "Immagine salvata come " << outputFilename2 << std::endl;
            std::cout << "Tempo di esecuzione: " << duration2 << " ms" << std::endl;

            if(k.isSeparable()){
                auto start3 = std::chrono::high_resolution_clock::now();
                STBImage result3 = separableConvolutionRGB(loadedImages[i], k);
                auto end3 = std::chrono::high_resolution_clock::now();
                auto duration3 = std::chrono::duration<double, std::milli>(end3 - start3).count();
                std::string originalFilename3 = std::filesystem::path(loadedImages[i].filename).filename().string();
                std::string outputFilename3 = "images/separable_convolution/" + originalFilename3;
                result1.saveImage(outputFilename3);
                std::cout << "Immagine salvata come " << outputFilename3 << std::endl;
                std::cout << "Tempo di esecuzione: " << duration3 << " ms" << std::endl;

                auto start4 = std::chrono::high_resolution_clock::now();
                STBImage result4 = separableConvolutionRGB_CUDA(loadedImages[i], k);
                auto end4 = std::chrono::high_resolution_clock::now();
                auto duration4 = std::chrono::duration<double, std::milli>(end4 - start4).count();
                std::string originalFilename4 = std::filesystem::path(loadedImages[i].filename).filename().string();
                std::string outputFilename4 = "images/separable_convolutionCUDA/" + originalFilename4;
                result1.saveImage(outputFilename4);
                std::cout << "Immagine salvata come " << outputFilename4 << std::endl;
                std::cout << "Tempo di esecuzione: " << duration4 << " ms" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }

    return 0;
}
