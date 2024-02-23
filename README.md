# Proyek Pertama

#### Disusun oleh : Rahman Wahabi Hasibuan

Ini adalah proyek pertama predictive analytics untuk memenuhi submission dicoding. Proyek ini membangun model machine learning yang dapat memprediksi emosi berdasarkan teks yg tersebar di sosmed

## Domain Proyek

### Latar Belakang

Emosi yang sesuai dengan waktu dan keadaan dapat mempengaruhi hasil aktivitas yang dilakukan oleh manusia. Emosi itu seringkali diungkapkan secara tidak langsung oleh manusia dan juga dipicu oleh peristiwa atau situasi tertentu. Teks menunjukkan pola peristiwa atau situasi yang menyebabkan emosi dapat diungkapkan dengan kata-kata secara terang (eksplisit). Teks juga menjadi sarana utama dalam komunikasi menggunakan komputer (_Computer-Mediated Communication_) seperti email, blog dan media sosial.

<br>

<div><img src="https://i.ibb.co/h29JPCn/collection-young-people-emotions-52683-18562-e16956101281601.jpg" width="1000"/></div>

<br>

Analisis sentimen digunakan untuk mengekstrasi
sebuah polaritas opini dari sebuah kata atau kalimat menjadi
sebuah polaritas positif, negatif atau netral. Dengan melakukan
analisis emosi, maka tidak hanya bentuk polaritas berupa
positif, netral atau negatif saja yang mampu dihasilkan, namun
juga bentuk emosional yang disampaikan oleh pengguna.
Analisis emosi dapat digunakan untuk meningkatkan layanan
pelanggan, sektor bisnis, pemasaran, dan juga sebagai ukuran
kinerja media sosial. Berdasarkan sifatnya emosi dapat dibedakan menjadi dua yaitu emosi positif dan emosi negatif. Yang termasuk emosi negatif antara lain _sadness_, _fear_, _anger_. Sedangkan yang termasuk dari emosi positif adalah _happy_ dan _love_.

Analisis emosi pada teks dari media sosial adalah bidang penelitian yang menarik banyak perhatian terutama untuk tujuan analisis emosi. Berdasarkan hasil survei yang telah dilakukan, dari tahun 2014-2017 ada 6 dari 10 paper yang dipublikasikan mengkaji tentang analisis emosi pada teks[1](https://jurnal.umnu.ac.id/index.php/juristik/article/download/365/115/969). Berdasarkan latar belakang peneliti sebelumnya penulis mengambil kesimpulan untuk metode klasifikasi yang digunakan, diantaranya _Logistic Regresion_, _Suport Vector Machine_, dan _LTSM_.

## Business Understanding
***
### Problem Statements
- Dalam era digital dan sosial media, pemahaman terhadap sentimen dan emosi yang terkandung dalam teks dari media sosial sangat penting bagi perusahaan untuk memahami persepsi publik terhadap merek atau produk mereka.
- Saat ini, perusahaan belum memiliki sistem yang efektif untuk menganalisis emosi dalam teks dari media sosial secara otomatis dan menyeluruh.

### Goals
- Mengembangkan sistem analisis emosi pada teks dari media sosial yang dapat mengidentifikasi dan mengklasifikasikan emosi secara akurat.
- Meningkatkan pemahaman terhadap persepsi publik terhadap merek atau produk melalui analisis emosi yang lebih mendalam.
- membuat model dan membandingkannya yang mana model yang terbaik

### Solution Statements
- Menggunakan tiga pendekatan klasifikasi yang berbeda, yaitu _Logistic Regression_, _Support Vector Machine_ dan _LSTM_, untuk mengklasifikasikan emosi dalam teks media sosial.
- Menggunakan _Classifictaion Report_ untuk mengevaluasi model saat testing dan accuracy saat train.

## Data Understanding
***
Dataset yang digunakan dalam proyek ini adalah PRDECT-ID Dataset adalah koleksi data ulasan produk Indonesia yang diberi label emosi dan sentimen. Data diambil dari salah satu e-commerce terbesar di Indonesia yaitu Tokopedia. Dataset ini berisi ulasan produk dari 29 kategori produk di Tokopedia yang menggunakan bahasa Indonesia. Setiap ulasan produk diberi label emosi, yaitu cinta, bahagia, marah, takut, atau sedih. Tim anotator melakukan proses pemberian label emosi dengan mengacu pada kriteria anotasi emosi yang telah dibuat oleh seorang ahli psikologi klinis. Atribut-atribut lain yang berkaitan dengan ulasan produk juga diambil, seperti Lokasi, Harga, Rating Keseluruhan, Jumlah Terjual, Jumlah Ulasan, dan Rating Pelanggan, untuk mendukung penelitian lebih lanjut. Dataset ini dapat diunduh di [Kaggle : House Rent Prediction Dataset](https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification/data).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 5400 sample dengan 11 fitur.
+ Dataset di bagian _Emotion_ bertipe int64 
+ Tidak ada missing value dalam dataset

### Variable - variable pada dataset

+ Category
+ Product Name
+ Location
+ Price
+ Overall Rating
+ Number Sold
+ Total Review
+ Customer Rating
+ Customer Review
+ Sentiment
+ Emotion

### Menampilkan informasi dari dataset
Pada bagian ini akan dipakai fungsi shape() dan value_counts() untuk mengetahui jumlah dataset dan sebaran dari label, informasi dari jumlah dan sebaran dapat dilihat pada tabel 1-2

Tabel 1. _Dataset Shape_. 
| DataFrame | Shape      |
| --------- | ---------- |
| df        | (16000, 2) |

Tabel 2. distribusi label pada **df**
| label    | jumlah |
| -------- | ------ |
| happy    | 1770   |
| sadness  | 1202   |
| anger    | 699    |
| fear     | 920    |
| love     | 809    |


### Mengecek missing value dan menangani jika ditemukan
Pada bagian ini digunakan fungsi `isnull().sum()` untuk tiap _DataFrame_. Saat dicek tidak ditemukan adanya _missing value_ pada _DataFrame_

| Category           0 |
| Product Name       0 |
| Location           0 |
| Price              0 |
| Overall Rating     0 |
| Number Sold        0 |
| Total Review       0 |
| Customer Rating    0 |
| Customer Review    0 |
| Sentiment          0 |
| Emotion            0 |
| dtype: int64         |

## Data Preparation
***
1. Normalisasi Teks:
- Lowercasing: Mengubah semua karakter teks menjadi huruf kecil untuk konsistensi.
- Remove Stop Words: Menghapus kata-kata umum yang tidak memiliki makna penting dalam pemrosesan teks.
- Lemmatisasi: Mengubah kata-kata menjadi bentuk dasar (lemah) untuk mengurangi variasi kata yang sama.
Alasan: Normalisasi teks membantu mengurangi variasi dan mempersiapkan teks untuk analisis lebih lanjut dengan membuat teks lebih konsisten dan mudah diproses.

2. Pembersihan Teks:
- Menghapus Angka: Menghilangkan angka dari teks karena angka sering kali tidak memberikan kontribusi signifikan terhadap analisis sentimen atau emosi.
- Menghapus Tanda Baca: Menghilangkan tanda baca dari teks karena tanda baca tidak selalu penting dalam analisis emosi.
- Menghapus URL: Menghapus URL dari teks karena URL tidak berkontribusi pada analisis emosi.
Alasan: Pembersihan teks membantu menyederhanakan teks dan menghilangkan informasi yang tidak relevan untuk analisis emosi.

3. Tokenisasi:
Mengubah teks menjadi urutan token atau kata-kata.
Alasan: Tokenisasi diperlukan untuk memisahkan teks menjadi bagian-bagian yang lebih kecil sehingga dapat diolah lebih lanjut, seperti dalam proses pembuatan model.

4. Padding:
Menambahkan padding pada teks untuk membuat semua teks memiliki panjang yang sama.
Alasan: Padding diperlukan karena model yang akan digunakan membutuhkan input dengan panjang yang sama, sehingga padding digunakan untuk membuat semua teks memiliki panjang yang seragam.

Teknik-teknik data preparation ini diperlukan untuk membersihkan, menyederhanakan, dan mempersiapkan teks dari media sosial sehingga dapat diolah dan dianalisis lebih lanjut untuk analisis emosi.

## Modeling
***
### Tahapan dan Parameter Pemodelan
1. TF-IDF Embedding: 
- Digunakan untuk mengubah teks ke dalam representasi vektor numerik.
- Parameter: Beberapa parameter yang dapat disesuaikan dalam TF-IDF, seperti penggunaan stop words, n-grams, dll.

2. Logistic Regression:
- Logistic Regression adalah sebuah metode analisis statistik yang membangun sebuah model statistik untuk menggambarkan hubungan antara sebuah variabel hasil yang bersifat biner atau dikotomis (ya/tidak) dengan satu atau lebih variabel prediktor atau penjelas. Model ini menggunakan fungsi logit untuk memodelkan peluang dari sebuah kejadian biner[2](https://link.springer.com/referenceworkentry/10.1007/978-3-319-69909-7_1689-2).
- Parameter: Solver ('liblinear'), random_state.
- Kelebihan: Mudah diimplementasikan, memiliki interpretasi yang baik, efisien untuk dataset besar.
- Kekurangan: Tidak dapat menangani hubungan yang kompleks.

3. Support Vector Machine:
- Support Vector Machine (SVM) adalah sebuah metode pembelajaran terbimbing yang dapat digunakan untuk klasifikasi, regresi, estimasi densitas, deteksi anomali, dan aplikasi lainnya. Dalam kasus klasifikasi dua kelas yang paling sederhana, SVM mencari sebuah bidang hiper yang memisahkan dua kelas data dengan margin sebesar-besarnya. Hal ini menghasilkan akurasi generalisasi yang baik pada data yang belum dilihat, dan mendukung metode optimisasi khusus yang memungkinkan SVM untuk belajar dari jumlah data yang besar[3](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_804).
- Parameter: Kernel (default='rbf'), random_state.
- Kelebihan: Efektif dalam ruang fitur dengan banyak dimensi, dapat menangani data non-linier.
- Kekurangan: Membutuhkan waktu komputasi yang lama untuk dataset besar.

4. LSTM:
- LSTM adalah singkatan dari Long Short-Term Memory, yaitu sebuah arsitektur jaringan saraf berulang (RNN) yang dirancang untuk mengatasi masalah gradien yang menghilang dan meledak pada RNN konvensional. Berbeda dengan jaringan saraf maju, RNN memiliki koneksi siklik yang membuatnya kuat untuk memodelkan urutan[4](https://link.springer.com/article/10.1007/s10462-020-09838-1).
- Parameter: Jumlah unit LSTM, dropout rate, batch size, jumlah epoch.
- Kelebihan: Efektif untuk memahami konteks dalam teks panjang, dapat menangani data sekuen.
- Kekurangan: Membutuhkan waktu dan sumber daya komputasi yang besar, rentan terhadap overfitting.

### Proses Improvement dengan Hyperparameter Tuning
Pada model LSTM, kami melakukan improvement dengan melakukan hyperparameter tuning. Beberapa langkah yang dilakukan adalah:

1. Penambahan layer LSTM dan penyesuaian jumlah unit LSTM.
2. Penyesuaian dropout rate untuk mengurangi overfitting.
3. Penggunaan BatchNormalization untuk mempercepat konvergensi model.
4. Penggunaan early stopping dan learning rate reduction untuk menghindari overfitting.

Proses improvement ini dilakukan untuk meningkatkan performa model LSTM dalam menganalisis sentimen emosi pada data review pelanggan.

<br>

<div><img src="https://i.ibb.co/HhRvybW/download.png" width="300"/></div>

<br>

### Pemilihan Model Terbaik
Dari ketiga model yang digunakan, model LSTM dipilih sebagai model terbaik karena mampu memberikan kinerja yang lebih baik dalam memahami konteks teks panjang dan menangani data sekuen dengan baik. LSTM memiliki kelebihan dalam menangani data yang bersifat sekuensial, seperti review pelanggan, dan dapat mengatasi masalah vanishing gradient yang sering terjadi pada model RNN tradisional.

## Evaluasi
***
Pada proyek ini menggunakan model deep learning bertipe klasifikasi yang berarti jika mendekati 100% accuracy, performanya bagus, sedangkan jika dibawah 75%, maka performanya jelek. Metrik yang akan kita pakai pada prediksi ini adalah Accuracy, metrik ini menghitung persentase prediksi yang benar dari total prediksi yang dilakukan. Accuracy ditulis dalam rumus berikut:

$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$   

Metrik-metrik lain selain accuracy yang akan digunakan adalah precision, recall, dan f1-score yang dihasilkan dengan fungsi classification_report yang tersedia dari library sklearn.

Presisi digunakan untuk mengukur seberapa akurat sebuah model ketika memberikan prediksi terhadap suatu kelas/tujuan. Presisi ditulis dalam rumus berikut:

$\text{Precision} = \frac{TP}{TP+FP}$

Recall digunakan untuk mengukur kemampuan model untuk memprediksi kelas True Positive. Recall ditulis dalam rumus berikut:

$\text{Recall} = \frac{TP}{TP+FN}$  

F1-Score digunakan untuk mencari titik seimbang antara Presisi dan Recall, F1-Score ditulis dalam rumus berikut:

$\text{F1-Score} = \frac{2 \* Precision \* Recall}{Precision+Recall}$    

Dengan:

- TP: True Positive
- TN: True Negative
- FP: False Positive
- FN: False Negative

Berikut hasil evaluasi pada proyek ini :
+ Akurasi
  | model               | accuracy |
  |---------------------|----------|
  | LTSM                | 0.98     |
  | SVM                 | 0.97     |
  | Logistic Regression | 0.82     |

    <div><img src="https://i.ibb.co/RTDt9L4/download-1.png" width="300"/></div>

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma LTSM memiliki akurasi lebih tinggi tinggi dibandingkan algoritma lainnya dalam proyek ini.

+ Presisi, Recall, F1-Score
- Logistic Regression

              precision    recall  f1-score   support

       Anger       0.90      0.67      0.77       675
        Fear       0.81      0.76      0.78       892
       Happy       0.81      0.95      0.87      1753
        Love       0.87      0.62      0.72       800
       Sadness       0.79      0.90      0.84      1185

- SVM

              precision    recall  f1-score   support

       Anger       0.98      0.95      0.97       675
        Fear       0.98      0.97      0.98       892
       Happy       0.97      0.99      0.98      1753
        Love       0.98      0.94      0.96       800
       Sadness       0.96      0.99      0.97      1185

- LTSM

              precision    recall  f1-score   support

           0       0.98      0.98      0.98       675
           1       0.99      0.96      0.97       892
           2       0.99      0.99      0.99      1753
           3       0.99      0.97      0.98       800
           4       0.96      0.99      0.98      1185

Dari hasil evaluasi di atas, dapat disimpulkan beberapa hal:

1. Akurasi Model: Model LSTM memiliki akurasi tertinggi yaitu 0.98, diikuti oleh SVM dengan akurasi 0.97, dan Logistic Regression dengan akurasi 0.82. Hal ini menunjukkan bahwa model LSTM lebih baik dalam melakukan klasifikasi emosi pada data review pelanggan dibandingkan dengan model lainnya.

2. Presisi, Recall, F1-Score:
- Logistic Regression: Presisi tertinggi untuk kelas Happy (0.81), Recall tertinggi untuk kelas Love (0.87), dan F1-Score tertinggi untuk kelas Happy (0.87).
- SVM: Presisi tertinggi untuk kelas Happy (0.97), Recall tertinggi untuk kelas Sadness (0.99), dan F1-Score tertinggi untuk kelas Fear (0.98).
- LSTM: Presisi, Recall, dan F1-Score tertinggi untuk semua kelas (0.98), menunjukkan kemampuan yang sangat baik dalam mengklasifikasikan setiap kelas emosi.

Referensi:  
  [1]    
  [Fadjeri A, Hidayat K, Handayani DR. Deteksi Emosi pada Teks menggunakan Algoritma Naïve Bayes. J Ris Teknol dan Komput. 2021;1(2):1-4. doi:10.53863/juristik.v1i02.365.](https://jurnal.umk.ac.id/index.php/simet/article/view/3487)

  [2]  
  [Cheung AKL. Encyclopedia of Quality of Life and Well-Being.; 2021.](https://link.springer.com/referencework/10.1007/978-3-319-69909-7)

  [3]  
  [Sammut C, Webb GI. Front Matter. Encycl Mach Learn. Published online 2011.](https://link.springer.com/referencework/10.1007/978-0-387-30164-8)

  [4]
  [Van Houdt G, Mosquera C, Nápoles G. A review on the long short-term memory model. Artif Intell Rev. 2020;53(8):5929-5955. doi:10.1007/s10462-020-09838-1.](https://link.springer.com/article/10.1007/s10462-020-09838-1)