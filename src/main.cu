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
std::ifstream conf_file("settings/config.json");
const json CONFIG = json::parse(std::ifstream("settings/config.json"));

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct STBImage {
    int width{0}, height{0}, channels{0};
    uint8_t *image_data{nullptr};
    std::string filename{};

    // Costruttore di default
    STBImage() = default;

    // Distruttore
    ~STBImage() {
        if (image_data) {
            stbi_image_free(image_data);
        }
    }

    // Costruttore di copia
    STBImage(const STBImage& other) {
        width = other.width;
        height = other.height;
        channels = other.channels;
        filename = other.filename;
        size_t size = width * height * channels;
        if (other.image_data) {
            image_data = (uint8_t*)malloc(size);
            std::memcpy(image_data, other.image_data, size);
        }
    }

    // Assegnamento di copia
    STBImage& operator=(const STBImage& other) {
        if (this == &other) return *this;
        free(image_data);
        width = other.width;
        height = other.height;
        channels = other.channels;
        filename = other.filename;
        size_t size = width * height * channels;
        if (other.image_data) {
            image_data = (uint8_t*)malloc(size);
            std::memcpy(image_data, other.image_data, size);
        } else {
            image_data = nullptr;
        }
        return *this;
    }

    // Costruttore di spostamento
    STBImage(STBImage&& other) noexcept {
        width = other.width;
        height = other.height;
        channels = other.channels;
        image_data = other.image_data;
        filename = std::move(other.filename);
        other.image_data = nullptr;
    }

    // Assegnamento di spostamento
    STBImage& operator=(STBImage&& other) noexcept {
        if (this == &other) return *this;
        free(image_data);
        width = other.width;
        height = other.height;
        channels = other.channels;
        image_data = other.image_data;
        filename = std::move(other.filename);
        other.image_data = nullptr;
        return *this;
    }

    // Carica immagine
    bool loadImage(const std::string &name) {
        if (image_data) {
            stbi_image_free(image_data);
        }
        image_data = stbi_load(name.c_str(), &width, &height, &channels, 3);
        if (channels == 4)
            channels = 3;
        if (!image_data)
            return false;
        filename = name;
        return true;
    }

    void saveImage(const std::string &newName) const {
        stbi_write_jpg(newName.c_str(), width, height, channels, image_data, width);
    }

    void initializeRGB(int w, int h) {
        if (image_data) {
            free(image_data);
        }
        width = w;
        height = h;
        channels = 3;
        image_data = (uint8_t*)malloc(width * height * channels);
        std::memset(image_data, 0, width * height * channels); 
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
        float tol = 1e-6f * singularValues(0);  // tolleranza relativa

        return singularValues(0) > tol && singularValues.tail(singularValues.size() - 1).array().abs().maxCoeff() < tol;
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

        // Tolleranza relativa basata sul valore singolare massimo
        float tol = 1e-6f * singularValues(0);

        // Controlliamo se il primo valore singolare è significativo e gli altri sono (quasi) zero
        if (singularValues(0) < tol || singularValues.tail(singularValues.size() - 1).array().abs().maxCoeff() > tol) {
            return false;
        }

        // Otteniamo i vettori singolari principali
        Eigen::VectorXf u = svd.matrixU().col(0);  // Primo vettore colonna
        Eigen::VectorXf v = svd.matrixV().col(0);  // Primo vettore colonna (vettore riga nella matrice originale)

        // Copiamo i vettori in std::vector
        vertical.assign(u.data(), u.data() + u.size());
        horizontal.assign(v.data(), v.data() + v.size());

        // Scala i vettori per la radice quadrata del primo valore singolare
        float sigma = std::sqrt(singularValues(0));
        for (float& val : vertical) val *= sigma;
        for (float& val : horizontal) val *= sigma;

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

std::vector<std::vector<float>> createKernel(int size, const std::string& type) {
    if (size % 2 == 0 || size < 3) {
        throw std::invalid_argument("La dimensione del kernel deve essere dispari e >= 3.");
    }

    std::vector<std::vector<float>> kernel(size, std::vector<float>(size, 0.0f));
    int mid = size / 2;

    if (type == "sharpen") {
        // Tutti -1
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                kernel[i][j] = -1.0f;

        // Centro molto positivo per accentuare i bordi
        kernel[mid][mid] = static_cast<float>(size * size);  // Es: 121 per 11x11
    }
    else if (type == "gaussian") {
        std::vector<int> pascal(size, 1);
        for (int i = 1; i < size; ++i)
            pascal[i] = pascal[i - 1] * (size - i) / i;

        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                kernel[i][j] = static_cast<float>(pascal[i] * pascal[j]);
    }
    else {
        throw std::invalid_argument("Tipo kernel non supportato. Usa 'gaussian' o 'sharpen'.");
    }

    return kernel;
}


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

// Dummy CUDA kernel per warmup
__global__ void dummyKernel() {
}

void cudaWarmup() {
    //Lancia un piccolo kernel per inizializzare la GPU
    auto start = std::chrono::high_resolution_clock::now();
    dummyKernel<<<1, 1>>>();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    std::cout << "Tempo di esecuzione CUDA warmup: " << duration << " ms" << std::endl;
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

    if (x >= width || y >= height) return;

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

// Funzione per fare image processing tramite convoluzione di un kernel con CUDA
STBImage convolveRGB_CUDA(const STBImage &img, const Kernel &kernel, int block_size_x, int block_size_y) {
    std::vector<float> h_allKernel(kernel.size * kernel.size);
    for (int i = 0; i < kernel.size; i++)
        for (int j = 0; j < kernel.size; j++)
            h_allKernel[i * kernel.size + j] = kernel.matrix[i][j];
    
    STBImage outputImg;
    outputImg.initializeRGB(img.width, img.height);
    
    int imageSize = img.width * img.height * 3;
    
    uint8_t *d_input, *d_output;
    float *d_kernel;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel.size * kernel.size * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, img.image_data, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_allKernel.data(), kernel.size * kernel.size * sizeof(float), cudaMemcpyHostToDevice));


    dim3 blockSize(block_size_x, block_size_y);
    dim3 gridSize((img.width + blockSize.x - 1) / blockSize.x, (img.height + blockSize.y - 1) / blockSize.y);
    
    convolveKernelRGB<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, img.width, img.height, kernel.size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(outputImg.image_data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));

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

    return outputImg;
}

// CUDA Kernel per la convoluzione orizzontale (primo passaggio)
__global__ void convolutionHorizontalKernelRGB(const unsigned char *input, unsigned char *tempOutput, const float *hKernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenter = kernelSize / 2;
    int pixelIndex = (y * width + x) * 3;
    float sum[3] = {0};

    if (x >= width || y >= height) return;

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

// CUDA Kernel per la convoluzione verticale (second pass)
__global__ void convolutionVerticalKernelRGB(const unsigned char *tempInput, unsigned char *output, const float *vKernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenter = kernelSize / 2;
    int pixelIndex = (y * width + x) * 3;
    float sum[3] = {0};

    if (x >= width || y >= height) return;

    
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

// Funzione per fare Image Processing tramite convoluzione di un kernel separabile con CUDA
STBImage separableConvolutionRGB_CUDA(const STBImage &img, Kernel &kernel, int block_size_x, int block_size_y) {
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

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempOutput, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hKernel, kernel.size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_vKernel, kernel.size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, img.image_data, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_hKernel, h.data(), kernel.size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vKernel, v.data(), kernel.size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(block_size_x, block_size_y);
    dim3 gridSize((img.width + blockSize.x - 1) / blockSize.x, (img.height + blockSize.y - 1) / blockSize.y);

    convolutionHorizontalKernelRGB<<<gridSize, blockSize>>>(d_input, d_tempOutput, d_hKernel, img.width, img.height, kernel.size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    convolutionVerticalKernelRGB<<<gridSize, blockSize>>>(d_tempOutput, d_output, d_vKernel, img.width, img.height, kernel.size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(outputImg.image_data, d_output, img.width * img.height * img.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_tempOutput));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_hKernel));
    CHECK_CUDA_ERROR(cudaFree(d_vKernel));

    return outputImg;
}
//<------------------------------------------------------------------------>

std::string format_double(double value, int precision = 4) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

int main(){

    createPath("images/convolution");
    createPath("images/convolutionCUDA");
    createPath("images/separable_convolution");
    createPath("images/separable_convolutionCUDA");
    createPath("results");

    cudaWarmup();

    std::string dataset_name = CONFIG["dataset_name"];
    int kernel_size = CONFIG["kernel_size"];
    std::string kernel_type = CONFIG["kernel_type"];
    bool kernel_normalize = CONFIG["kernel_normalize"];
    std::vector<std::pair<int, int>> block_combinations = {
        {4, 4},
        {8, 4},
        {8, 8},
        {16,8},
        {16, 16},
        {32, 8},
        {32, 16},
        {32, 32}
    };

    double convolution_seq_mean = 0;
    double convolution_cuda_mean = 0;
    double separable_convolution_seq_mean = 0;
    double separable_convolution_cuda_mean = 0;
    std::vector<double> convolution_seq_times;
    std::vector<double> convolution_cuda_times;
    std::vector<double> separable_convolution_seq_times;
    std::vector<double> separable_convolution_cuda_times;

    auto calculateMeanTime = [](const std::vector<double> &test_times, double &mean_time) {
        double sum = std::accumulate(test_times.begin(), test_times.end(), 0.0);
        mean_time = sum / test_times.size();
    };

    std::string filePathCSV = "results/" + kernel_type + "_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size) + "/";
    createPath(filePathCSV);

    // Nomi file CSV
    std::string csv_speedup_filename = filePathCSV + "csv_speedup_" + kernel_type + "_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size) + ".csv";
    std::string csv_times_filename = filePathCSV + "csv_times_" + kernel_type + "_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size) + ".csv";
    std::string txt_results_filename = filePathCSV + "times_summary.txt";

    {
        std::ofstream csv_speedup(csv_speedup_filename, std::ios::trunc);
        std::ofstream csv_times(csv_times_filename, std::ios::trunc);
        std::ofstream txt_results(txt_results_filename, std::ios::trunc);

        if (!csv_speedup || !csv_times || !txt_results) {
            std::cerr << "Errore nell'apertura dei file CSV o TXT." << std::endl;
            return 1;
        }

        csv_speedup << "Block_Size;Convolution;Separable\n";
        csv_times   << "Block_Size;Convolution;Separable\n";
        txt_results << "== Riepilogo Tempi ==\n";
    }

    // Apertura in append per tutto il resto dell'esecuzione
    std::ofstream csv_speedup(csv_speedup_filename, std::ios::app);
    std::ofstream csv_times(csv_times_filename, std::ios::app);
    std::ofstream txt_results(txt_results_filename, std::ios::app);

    std::string dataset_path = "images/" + dataset_name;
    std::cout << "Caricamento immagini dal dataset: " << dataset_path << std::endl;

    std::vector<STBImage> loadedImages = loadImages(dataset_path);
    if (loadedImages.empty()) {
        std::cerr << "Nessuna immagine trovata nella cartella " << dataset_path << "'." << std::endl;
        return 1;
    }
    std::cout << "Totale immagini caricate: " << loadedImages.size() << std::endl;

    try {
        std::vector<std::vector<float>> kernel_vec = createKernel(kernel_size, kernel_type);
        if (kernel_vec.empty()) {
            throw std::runtime_error("Errore nella creazione del kernel. Assicurati che la dimensione sia dispari e >= 3.");
        }
        
        Kernel k(kernel_size, kernel_vec, kernel_normalize);
        k.print();
        std::cout << "il kernel e' separabile? " << (k.isSeparable() ? "Si" : "No") << std::endl;

        std::cout << "\n<==== Test sequenziale ====>" << std::endl;
        for (size_t i = 0; i < loadedImages.size(); i++) {
            std::cout << "<-------------------------------------------------------->" << std::endl;

            // Convoluzione sequenziale CPU
            auto start_cpu = std::chrono::high_resolution_clock::now();
            STBImage result_cpu = convolveRGB(loadedImages[i], k);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            double duration_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
            convolution_seq_times.push_back(duration_cpu);
            result_cpu.saveImage("images/convolution/" + std::filesystem::path(loadedImages[i].filename).filename().string());
            std::cout << "Immagine salvata (CPU convoluzione): images/convolution/" << std::filesystem::path(loadedImages[i].filename).filename().string() << std::endl;
            std::cout << "Durata CPU convoluzione: " << duration_cpu << " ms" << std::endl;

            // Convoluzione separabile CPU (se possibile)
            if (k.isSeparable()) {
                auto start_sep_cpu = std::chrono::high_resolution_clock::now();
                STBImage result_sep_cpu = separableConvolutionRGB(loadedImages[i], k);
                auto end_sep_cpu = std::chrono::high_resolution_clock::now();
                double duration_sep_cpu = std::chrono::duration<double, std::milli>(end_sep_cpu - start_sep_cpu).count();
                separable_convolution_seq_times.push_back(duration_sep_cpu);
                result_sep_cpu.saveImage("images/separable_convolution/" + std::filesystem::path(loadedImages[i].filename).filename().string());
                std::cout << "Immagine salvata (CPU convoluzione separabile): images/separable_convolution/" << std::filesystem::path(loadedImages[i].filename).filename().string() << std::endl;
                std::cout << "Durata CPU convoluzione separabile: " << duration_sep_cpu << " ms" << std::endl;
            }
            std::cout << "<-------------------------------------------------------->" << std::endl;
        }

        calculateMeanTime(convolution_seq_times, convolution_seq_mean);
        if (k.isSeparable()) {
            calculateMeanTime(separable_convolution_seq_times, separable_convolution_seq_mean);
        }

        // Scrittura speedup
        csv_speedup << "SEQ" << ";"
                    << format_double(1) << ";"
                    << format_double(1) << "\n";

        // Scrittura tempi
        csv_times << "SEQ" << ";"
                  << format_double(convolution_seq_mean) << ";"
                  << format_double(separable_convolution_seq_mean) << "\n";

        convolution_seq_times.clear();
        separable_convolution_seq_times.clear();

        for (auto [bx, by] : block_combinations) {
            std::cout << "\n<==== Test parallelo con block_size_x = " << bx << ", block_size_y = " << by << " ====>" << std::endl;

            for (size_t i = 0; i < loadedImages.size(); i++) {
                std::cout << "<-------------------------------------------------------->" << std::endl;

                // Convoluzione CUDA
                auto start_cuda = std::chrono::high_resolution_clock::now();
                STBImage result_cuda = convolveRGB_CUDA(loadedImages[i], k, bx, by);
                auto end_cuda = std::chrono::high_resolution_clock::now();
                double duration_cuda = std::chrono::duration<double, std::milli>(end_cuda - start_cuda).count();
                convolution_cuda_times.push_back(duration_cuda);
                result_cuda.saveImage("images/convolutionCUDA/" + std::filesystem::path(loadedImages[i].filename).filename().string());
                std::cout << "Immagine salvata (CUDA convoluzione): images/convolutionCUDA/" << std::filesystem::path(loadedImages[i].filename).filename().string() << std::endl;
                std::cout << "Durata CUDA convoluzione: " << duration_cuda << " ms" << std::endl;
                

                // Convoluzione separabile CUDA (se possibile)
                if (k.isSeparable()) {
                    auto start_sep_cuda = std::chrono::high_resolution_clock::now();
                    STBImage result_sep_cuda = separableConvolutionRGB_CUDA(loadedImages[i], k, bx, by);
                    auto end_sep_cuda = std::chrono::high_resolution_clock::now();
                    double duration_sep_cuda = std::chrono::duration<double, std::milli>(end_sep_cuda - start_sep_cuda).count();
                    separable_convolution_cuda_times.push_back(duration_sep_cuda);
                    result_sep_cuda.saveImage("images/separable_convolutionCUDA/" + std::filesystem::path(loadedImages[i].filename).filename().string());
                    std::cout << "Immagine salvata (CUDA convoluzione separabile): images/separable_convolutionCUDA/" << std::filesystem::path(loadedImages[i].filename).filename().string() << std::endl;
                    std::cout << "Durata CUDA convoluzione separabile: " << duration_sep_cuda << " ms" << std::endl;
                }

                std::cout << "<-------------------------------------------------------->" << std::endl;
            }

            // Calcolo medie per questa configurazione di block size
            calculateMeanTime(convolution_cuda_times, convolution_cuda_mean);
            if (k.isSeparable()) {
                calculateMeanTime(separable_convolution_cuda_times, separable_convolution_cuda_mean);
            }

            // Scrittura speedup
            csv_speedup << "("+std::to_string(bx)+","+std::to_string(by)+")" << ";"
                        << format_double(convolution_seq_mean/convolution_cuda_mean) << ";"
                        << format_double(separable_convolution_seq_mean/separable_convolution_cuda_mean) << "\n";

            // Scrittura tempi
            csv_times << "("+std::to_string(bx)+","+std::to_string(by)+")" << ";"
                    << format_double(convolution_cuda_mean) << ";"
                    << format_double(separable_convolution_cuda_mean) << "\n";

            std::cout << "==> Risultati per block_size_x = " << bx << ", block_size_y = " << by << std::endl;
            std::cout << "   Seq CPU:         " << convolution_seq_mean << " ms" << std::endl;
            std::cout << "   CUDA:            " << convolution_cuda_mean << " ms" << std::endl;
            std::cout << "   Sep Seq CPU:     " << separable_convolution_seq_mean << " ms" << std::endl;
            std::cout << "   Sep CUDA:        " << separable_convolution_cuda_mean << " ms" << std::endl << std::endl;

            txt_results << "==> Risultati per block_size_x = " << bx << ", block_size_y = " << by << std::endl;
            txt_results << "   Seq CPU:         " << convolution_seq_mean << " ms" << std::endl;
            txt_results << "   CUDA:            " << convolution_cuda_mean << " ms" << std::endl;
            txt_results << "   Sep Seq CPU:     " << separable_convolution_seq_mean << " ms" << std::endl;
            txt_results << "   Sep CUDA:        " << separable_convolution_cuda_mean << " ms" << std::endl << std::endl;


            // Pulisci i vettori per la prossima configurazione
            convolution_cuda_times.clear();
            separable_convolution_cuda_times.clear();
        }
        csv_speedup.close();
        csv_times.close();
        txt_results.close();

    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
        csv_speedup.close();
        csv_times.close();
        txt_results.close();
    }

    return 0;
}
