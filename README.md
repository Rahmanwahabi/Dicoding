# Proyek Pertama

#### Disusun oleh : Rahman Wahabi Hasibuan

Ini adalah proyek pertama _predictive analytics_ untuk memenuhi submission dicoding. Proyek ini membangun model machine learning yang dapat memprediksi emosi berdasarkan teks yg tersebar di sosmed

## Domain Proyek

### Latar Belakang

Emosi yang sesuai dengan waktu dan keadaan dapat mempengaruhi hasil aktivitas yang dilakukan oleh manusia. Emosi itu seringkali diungkapkan secara tidak langsung oleh manusia dan juga dipicu oleh peristiwa atau situasi tertentu. Teks menunjukkan pola peristiwa atau situasi yang menyebabkan emosi dapat diungkapkan dengan kata-kata secara terang (eksplisit). Teks juga menjadi sarana utama dalam komunikasi menggunakan komputer (_Computer-Mediated Communication_) seperti email, blog dan media sosial.

<br>

<div><img src="https://i.ibb.co/h29JPCn/collection-young-people-emotions-52683-18562-e16956101281601.jpg" width="1000"/></div>

Gambar 1. Tampilan Emosi Manusia

<br>

Analisis sentimen digunakan untuk mengekstrasi sebuah polaritas opini dari sebuah kata atau kalimat menjadi sebuah polaritas positif, negatif atau netral. Dengan melakukan analisis emosi, maka tidak hanya bentuk polaritas berupa positif, netral atau negatif saja yang mampu dihasilkan, namun juga bentuk emosional yang disampaikan oleh pengguna. Analisis emosi dapat digunakan untuk meningkatkan layanan
pelanggan, sektor bisnis, pemasaran, dan juga sebagai ukuran kinerja media sosial. Berdasarkan sifatnya emosi dapat dibedakan menjadi dua yaitu emosi positif dan emosi negatif. Yang termasuk emosi negatif antara lain _sadness_, _fear_, _anger_. Sedangkan yang termasuk dari emosi positif adalah _happy_ dan _love_.

Analisis emosi pada teks dari media sosial adalah bidang penelitian yang menarik banyak perhatian terutama untuk tujuan analisis emosi. Berdasarkan hasil survei yang telah dilakukan, dari tahun 2014-2017 ada 6 dari 10 paper yang dipublikasikan mengkaji tentang analisis emosi pada teks[1](https://jurnal.umnu.ac.id/index.php/juristik/article/download/365/115/969). Berdasarkan latar belakang peneliti sebelumnya penulis mengambil kesimpulan untuk metode klasifikasi yang digunakan, diantaranya _Logistic Regresion_, _Suport Vector Machine_, dan _LSTM_.

### Manfaat bagi perusahaan atau industri tertentu:

- Analisis emosi dapat membantu perusahaan dalam mengukur kepuasan pelanggan, loyalitas merek, dan preferensi produk. Dengan mengetahui emosi pelanggan terhadap produk atau layanan tertentu, perusahaan dapat meningkatkan kualitas, menyesuaikan strategi pemasaran, dan memberikan layanan yang lebih baik. Misalnya, sebuah perusahaan otomotif dapat menggunakan analisis emosi untuk mengetahui emosi pelanggan terhadap fitur, desain, atau harga mobil yang mereka jual, dan kemudian melakukan perbaikan atau penawaran yang sesuai[2](https://link.springer.com/article/10.1007/s13278-021-00776-6).

- Analisis emosi dapat membantu perusahaan dalam mengidentifikasi dan mengelola krisis reputasi, risiko, dan peluang. Dengan memantau emosi publik terhadap isu-isu yang berkaitan dengan perusahaan, industri, atau kompetitor, perusahaan dapat mendeteksi adanya sentimen negatif, protes, atau keluhan yang dapat merusak citra atau bisnis mereka, dan kemudian meresponnya dengan cepat dan tepat. Misalnya, sebuah perusahaan penerbangan dapat menggunakan analisis emosi untuk mengetahui emosi penumpang terhadap kualitas layanan, keamanan, atau keterlambatan penerbangan, dan kemudian memberikan kompensasi, permintaan maaf, atau penjelasan yang sesuai[3](https://www.repustate.com/blog/sentiment-analysis-real-world-examples/).

- Analisis emosi dapat membantu perusahaan dalam mengembangkan produk atau layanan yang lebih inovatif, kreatif, dan sesuai dengan kebutuhan pasar. Dengan memahami emosi konsumen terhadap tren, gaya hidup, atau keinginan, perusahaan dapat menciptakan produk atau layanan yang lebih menarik, unik, dan relevan. Misalnya, sebuah perusahaan kosmetik dapat menggunakan analisis emosi untuk mengetahui emosi konsumen terhadap warna, aroma, atau tekstur produk kecantikan yang mereka gunakan, dan kemudian menciptakan produk kecantikan yang lebih sesuai dengan selera atau kebutuhan konsumen[4](https://blog.hootsuite.com/social-media-sentiment-analysis-tools/).

## Business Understanding
***
### Problem Statements
- Mengapa pemahaman terhadap sentimen dan emosi yang terkandung dalam teks dari media sosial sangat penting bagi perusahaan untuk memahami persepsi publik terhadap merek atau produk mereka?
- Bagaimana cara perusahaan untuk memiliki sistem yang efektif untuk menganalisis emosi dalam teks dari media sosial secara otomatis dan menyeluruh?
- Bagaiman cara membuat model yang akan digunakan untuk memprediksi?

### Goals
- Mengembangkan sistem analisis emosi pada teks dari media sosial yang dapat mengidentifikasi dan mengklasifikasikan emosi secara akurat.
- Meningkatkan pemahaman terhadap persepsi publik terhadap merek atau produk melalui analisis emosi yang lebih mendalam.
- membuat model dan membandingkannya yang mana model yang terbaik

### Solution Statements
- Menggunakan tiga pendekatan klasifikasi yang berbeda, yaitu _Logistic Regression_, _Support Vector Machine_ dan _LSTM_, untuk mengklasifikasikan emosi dalam teks media sosial.
- Menggunakan _Classifictaion Report_ untuk mengevaluasi model saat _testing_ dan _accuracy_ saat _train_.

### Studi kasus tentang bagaimana analisis emosi pada teks media sosial telah memberikan nilai tambah bagi perusahaan:

- Sebuah studi kasus yang dilakukan oleh Benrouba dan Boudour (2023)[5](https://www.semanticscholar.org/paper/Emotional-sentiment-analysis-of-social-media-for-Benrouba-Boudour/89682c3a79487a7af9895eb3067c80fc77804515), menunjukkan bagaimana analisis emosi dapat membantu meningkatkan kesehatan mental pengguna media sosial. Mereka mengusulkan sebuah sistem yang dapat memfilter konten media sosial yang dapat berdampak buruk secara emosional bagi pengguna, dengan mengklasifikasikan emosi dalam teks media sosial menjadi lima kategori dasar, yaitu cinta, bahagia, marah, takut, atau sedih. Sistem ini dapat membantu pengguna media sosial untuk menghindari konten yang dapat menimbulkan stres, depresi, atau kecemasan, dan memilih konten yang dapat meningkatkan suasana hati, motivasi, atau kepercayaan diri mereka.

- Sebuah studi kasus yang dilakukan oleh Nandwani dan Verma (2021)[2](https://link.springer.com/article/10.1007/s13278-021-00776-6), menunjukkan bagaimana analisis emosi dapat membantu perusahaan dalam mengukur kepuasan pelanggan, loyalitas merek, dan preferensi produk. Mereka mengusulkan sebuah sistem yang dapat mengidentifikasi dan mengklasifikasikan emosi dalam ulasan produk online, dengan menggunakan berbagai model emosi, seperti model _Plutchik_, model _Ekman_, dan model _Parrott_. Sistem ini dapat membantu perusahaan dalam mengetahui emosi pelanggan terhadap produk atau layanan tertentu, dan kemudian meningkatkan kualitas, menyesuaikan strategi pemasaran, dan memberikan layanan yang lebih baik.

- Sebuah studi kasus yang dilakukan oleh Alshahrani et al. (2021)[6](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255615), menunjukkan bagaimana analisis emosi dapat membantu perusahaan dalam mengidentifikasi dan mengelola krisis reputasi, risiko, dan peluang. Mereka mengusulkan sebuah sistem yang dapat memantau emosi publik terhadap isu-isu yang berkaitan dengan pandemi COVID-19, dengan menggunakan pendekatan _deep learning_ berbasis _LSTM_. Sistem ini dapat membantu perusahaan dalam mendeteksi adanya sentimen negatif, protes, atau keluhan yang dapat merusak citra atau bisnis mereka, dan kemudian meresponnya dengan cepat dan tepat.

## Data Understanding
***
Dataset yang digunakan dalam proyek ini adalah _PRDECT-ID_ Dataset adalah koleksi data ulasan produk Indonesia yang diberi label emosi dan sentimen. Data diambil dari salah satu _e-commerce_ terbesar di Indonesia yaitu Tokopedia. Dataset ini berisi ulasan produk dari 29 kategori produk di Tokopedia yang menggunakan bahasa Indonesia. Setiap ulasan produk diberi label emosi, yaitu cinta, bahagia, marah, takut, atau sedih. Tim anotator melakukan proses pemberian label emosi dengan mengacu pada kriteria anotasi emosi yang telah dibuat oleh seorang ahli psikologi klinis. Atribut-atribut lain yang berkaitan dengan ulasan produk juga diambil, seperti Lokasi, Harga, Rating Keseluruhan, Jumlah Terjual, Jumlah Ulasan, dan Rating Pelanggan, untuk mendukung penelitian lebih lanjut. Dataset ini dapat diunduh di [Kaggle : House Rent Prediction Dataset](https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification/data).

### Menampilkan informasi dari dataset

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 5400 sample dengan 11 fitur.
+ Dataset di bagian _Emotion_ bertipe int64 
+ Tidak ada missing value dalam dataset

### Variable - variable pada dataset

1. Category: 
- Variabel ini menunjukkan jenis atau kategori produk yang dijual, seperti pakaian, elektronik, buku, dll.
- Tipe: object

2. Product Name: 
- Variabel ini menunjukkan nama atau merek produk yang dijual, seperti sepatu Nike, laptop Asus, novel Harry Potter, dll.
- Tipe: object

+ Location: 
- Variabel ini menunjukkan lokasi atau kota tempat penjual berada, seperti Jakarta, Surabaya, Bandung, dll.
- Tipe: object

+ Price: 
- Variabel ini menunjukkan harga produk yang dijual, dalam satuan rupiah.
- Tipe: int64

+ Overall Rating:
- Variabel ini menunjukkan nilai rata-rata dari semua rating yang diberikan oleh pembeli kepada produk, dalam skala 1-5 bintang.
- Tipe: float64

+ Number Sold: 
- Variabel ini menunjukkan jumlah produk yang terjual, dalam satuan unit.
- Tipe: int64

+ Total Review: 
- Variabel ini menunjukkan jumlah ulasan atau testimoni yang diberikan oleh pembeli kepada produk, dalam satuan unit.
- Tipe: int64

+ Customer Rating: 
- Variabel ini menunjukkan nilai rating yang diberikan oleh pembeli kepada produk, dalam skala 1-5 bintang.
- Tipe: int64

+ Customer Review: 
- Variabel ini menunjukkan teks ulasan atau testimoni yang ditulis oleh pembeli tentang produk.
- Tipe: object

+ Sentiment: 
- Variabel ini menunjukkan sentimen atau sikap yang terkandung dalam teks ulasan atau testimoni, yaitu positif, negatif, atau netral.
- Tipe: object

+ Emotion: 
- Variabel ini menunjukkan emosi atau perasaan yang terkandung dalam teks ulasan atau testimoni, yaitu love, happy, anger, fear, dan sadness.
Tipe: object

Tabel 1. _Dataset Info_. 
| #  | Column          | Non-Null Count | Dtype   |
|----|-----------------|----------------|---------|
| 0  | Category        | 5400 non-null  | object  |
| 1  | Product Name    | 5400 non-null  | object  |
| 2  | Location        | 5400 non-null  | object  |
| 3  | Price           | 5400 non-null  | int64   |
| 4  | Overall Rating  | 5400 non-null  | float64 |
| 5  | Number Sold     | 5400 non-null  | int64   |
| 6  | Total Review    | 5400 non-null  | int64   |
| 7  | Customer Rating | 5400 non-null  | int64   |
| 8  | Customer Review | 5400 non-null  | object  |
| 9  | Sentiment       | 5400 non-null  | object  |
| 10 | Emotion         | 5400 non-null  | object  |

###  Informasi statistik pada masing-masing kolom

Output ini menunjukkan statistik deskriptif dari lima variabel numerik yang digunakan untuk analisis emosi pada teks dari media sosial. Output ini terdiri dari delapan baris yang masing-masing memiliki nama, seperti count, mean, std, dll. Output ini juga menunjukkan bahwa data memiliki 5400 observasi.

Tabel 2. _Dataset Describe_. 
|       | Price | Overall Rating | Number Sold |   Total Review | Customer Rating |
|------:|------:|---------------:|------------:|---------------:|----------------:|
| count | count |   5.400000e+03 | 5400.000000 |    5400.000000 |     5400.000000 |
|  mean |  mean |   2.386961e+05 |    4.854389 |   15961.951852 |     2168.645556 |
|  std  |  std  |   8.016337e+05 |    0.108259 |   74201.913338 |     2915.666035 |
|  min  |  min  |   1.000000e+02 |    4.100000 |       9.000000 |        4.000000 |
|  25%  |  25%  |   2.000000e+04 |    4.800000 |    1630.000000 |      576.000000 |
|  50%  |  50%  |   5.990000e+04 |    4.900000 |    3794.500000 |     1192.000000 |
|  75%  |  75%  |   1.500000e+05 |    4.900000 |    9707.000000 |     2582.000000 |
|  max  |  max  |   1.539900e+07 |    5.000000 | 1000000.000000 |    24500.000000 |

### Mengetahui jumlah dataset
Pada bagian ini akan dipakai fungsi _shape()_ dan value_counts() untuk mengetahui jumlah dataset dan sebaran dari label, informasi dari jumlah dan sebaran dapat dilihat pada tabel 1-2

Tabel 3. _Dataset Shape_. 
| DataFrame | Shape      |
| --------- | ---------- |
| df        | (16000, 2) |

Tabel 4. distribusi label pada **df**
| label    | jumlah |
| -------- | ------ |
| happy    | 1770   |
| sadness  | 1202   |
| anger    | 699    |
| fear     | 920    |
| love     | 809    |

<br>

<div><img src="https://i.ibb.co/J7YpLQd/download-2.png" width="300"/></div>

Gambar 2. Visualisasi distribusi label emosi

<br>

### Univariate Analysis
Univariate Analysis adalah menganalisis setiap fitur secara terpisah.

Fitur-fitur yang termasuk dalam numerical_features adalah fitur-fitur yang memiliki nilai numerik, yaitu price, Overall Rating, Number Sold, Total Review, dan Customer Rating. Fitur-fitur ini bisa digunakan untuk menganalisis atau mengukur kinerja, popularitas, atau kualitas dari produk.

<br>

<div><img src="https://i.ibb.co/zsWRVYT/download-1.png" width="300"/></div>

Gambar 3. Visualisasi Numerical Features

ini menunjukkan bahwa sebagian besar produk memiliki harga rendah dengan sedikit produk yang harga tinggi, mendapatkan rating keseluruhan tinggi, terjual dalam jumlah besar di awal tetapi kemudian penjualan menurun secara signifikan, mendapatkan review banyak di awal yang berkurang secara eksponensial seiring dengan bertambahnya jumlah review, dan memiliki rating pelanggan yang sangat rendah atau sangat tinggi dengan sedikit yang berada di antara keduanya.

<br>

Fitur-fitur yang termasuk dalam categorical_features adalah fitur-fitur yang memiliki nilai kategorik, yaitu Category, Product Name, Location, Sentiment, dan Emotion. Fitur-fitur ini bisa digunakan untuk mengelompokkan atau mengklasifikasikan produk berdasarkan jenis, nama, lokasi, sentimen, atau emosi yang terkait dengan produk.

contoh dari categorical_features yaitu category

Table 5. Fitur Category
|                          | jumlah sampel | persentase |
|-------------------------:|---------------|------------|
| Computers and Laptops    | 200           | 3.7        |
| Kitchen                  | 200           | 3.7        |
| Health                   | 200           | 3.7        |
| Beauty                   | 200           | 3.7        |
| Camera                   | 200           | 3.7        |
| Mother and Baby          | 200           | 3.7        |
| Phones and Tablets       | 200           | 3.7        |
| Gaming                   | 200           | 3.7        |
| Movies and Music         | 200           | 3.7        |
| Women's Fashion          | 200           | 3.7        |
| Men's Fashion            | 200           | 3.7        |
| Muslim Fashion           | 200           | 3.7        |
| Kids and Baby Fashion    | 200           | 3.7        |
| Electronics              | 200           | 3.7        |
| Books                    | 200           | 3.7        |
| Toys and Hobbies         | 200           | 3.7        |
| Sport                    | 200           | 3.7        |
| Other Products           | 200           | 3.7        |
| Carpentry                | 200           | 3.7        |
| Party Supplies and Craft | 200           | 3.7        |
| Body Care                | 200           | 3.7        |
| Animal Care              | 200           | 3.7        |
| Automotive               | 200           | 3.7        |
| Office & Stationery      | 200           | 3.7        |
| Food and Drink           | 200           | 3.7        |
| Household                | 200           | 3.7        |
| Tour and Travel          | 80            | 1.5        |
| Precious Metal           | 80            | 1.5        |
| Property                 | 40            | 0.7        |

hasil dari analisis fitur kategori ('Category') pada dataset yang dilakukan. 
jumlah sampel: merupakan jumlah total data yang terdapat pada setiap kategori.
persentase: merupakan persentase dari jumlah sampel terhadap total keseluruhan data dalam dataset.
Dari output tersebut, dapat dilihat distribusi jumlah sampel dan persentase untuk setiap kategori. Misalnya, kategori 'Computers and Laptops' memiliki 200 sampel yang menyumbang sekitar 3.7% dari total data. Begitu pula dengan kategori lainnya.

### multivariate Analysis

### Categorical Features

<br>

<div><img src="https://i.ibb.co/KF3BJ0x/download-3.png" width="300"/></div>

Gambar 4. Fitur kategori terhadap Price

Data tersebut berisi kategori, nama, deskripsi, harga, dan rating dari setiap produk. Untuk menghitung rata-rata harga relatif terhadap emosi, kita bisa menggunakan analisis sentimen untuk mengklasifikasikan deskripsi produk ke dalam lima emosi yang berbeda. Kemudian, kita bisa menghitung rata-rata dari harga produk yang termasuk dalam setiap emosi. Hasilnya bisa divisualisasikan dengan grafik batang.

<br>

### Numerical Features

### Categorical Features

<br>

<div><img src="https://i.ibb.co/WtKnfj5/download-4.png" width="300"/></div>

Gambar 5. Korelasi Price dengan fitur lainnya

- Harga produk memiliki korelasi positif dengan Overall Rating, artinya produk yang memiliki rating tinggi cenderung memiliki harga tinggi juga, dan sebaliknya.
- Harga produk memiliki korelasi negatif dengan Number Sold, artinya produk yang memiliki harga tinggi cenderung terjual dalam jumlah rendah, dan sebaliknya.
- Harga produk memiliki korelasi negatif dengan Total Review, artinya produk yang memiliki harga tinggi cenderung mendapatkan review sedikit, dan sebaliknya.

<br>

### Mengevaluasi skor korelasi

<br>

<div><img src="https://i.ibb.co/Pgpm0Yp/download-5.png" width="300"/></div>

Gambar 6. Mengevaluasi skor korelasi

Dari matriks korelasi di atas, dapat dilihat bahwa:

- Harga dan rating umum memiliki korelasi positif rendah (0.15). Hal ini menunjukkan bahwa ketika harga naik, maka rating umum juga naik sedikit.
- Harga dan jumlah produk yang dijual memiliki korelasi negatif rendah (-0.07). Hal ini menunjukkan bahwa ketika harga naik, maka jumlah produk yang dijual sedikit berkurang.
- Harga dan jumlah ulasan memiliki korelasi positif rendah (0.09). Hal ini menunjukkan bahwa ketika harga naik, maka jumlah ulasan juga sedikit naik.
- Rating umum dan jumlah produk yang dijual memiliki korelasi positif sedang (0.17). Hal ini menunjukkan bahwa ketika rating umum naik, maka jumlah produk yang dijual juga naik sedikit.
- Rating umum dan jumlah ulasan memiliki korelasi positif sedang (0.2). Hal ini menunjukkan bahwa ketika rating umum naik, maka jumlah ulasan juga naik sedikit.
- Jumlah produk yang dijual dan jumlah ulasan memiliki korelasi negatif sedang (-0.21). Hal ini menunjukkan bahwa ketika jumlah produk yang dijual naik, maka jumlah ulasan sedikit berkurang.
- Jumlah produk yang dijual dan rating customer memiliki korelasi negatif sedang (-0.6). Hal ini menunjukkan bahwa ketika jumlah produk yang dijual naik, maka rating customer sedikit turun.
- Jumlah ulasan dan rating customer memiliki korelasi negatif sangat kuat (-0.8). Hal ini menunjukkan bahwa ketika jumlah ulasan naik, maka rating customer sedikit turun.

<br>

## Data Preparation
***
Pada tahap ini, melakukan beberapa teknik data preparation untuk membersihkan, menyederhanakan, dan mempersiapkan teks dari media sosial sehingga dapat diolah dan dianalisis lebih lanjut untuk analisis emosi. Disini tidak menuliskan kode pada laporan ini, karena kami sudah menuliskannya pada notebook. Disini hanya menjelaskan cara kerja dan alasan dari setiap teknik yang akan digunakan.

### Mengecek missing value dan menangani jika ditemukan
Pada bagian ini digunakan fungsi `isnull().sum()` untuk tiap _DataFrame_. Saat dicek tidak ditemukan adanya _missing value_ pada _DataFrame_

Table 6. Evaluasi hasil missing value di tiap dataframe
| dtype                 | int64  |
| --------------------- | ------ |
| Category              | 0      |
| Product Name          | 0      |
| Location              | 0      |
| Price                 | 0      |
| Overall Rating        | 0      |
| Number Sold           | 0      |
| Total Review          | 0      |
| Customer Rating       | 0      |
| Customer Review       | 0      |
| Sentiment             | 0      |
| Emotion               | 0      |


### Menghitung jumlah _stopwords_ dalam data

Menghitung jumlah _stopwords_ dalam data dapat berguna untuk beberapa tujuan, seperti:

- Mengurangi ukuran data dan waktu pemrosesan dengan menghapus _stopwords_ yang tidak relevan.

- Meningkatkan kinerja model analisis teks dengan memfokuskan pada kata-kata yang memiliki makna atau informasi penting.

- Mengetahui karakteristik data dan gaya penulisan dengan melihat frekuensi dan distribusi _stopwords_.

<br>

<div><img src="https://i.ibb.co/z4hMGJt/download-3.png" width="300"/></div>

Gambar 7. Visualisasi distribusi kata-kata yang merupakan _stopwords_

<br>

1. Normalisasi Teks: Teknik ini bertujuan untuk membuat teks lebih konsisten dan mudah diproses dengan mengurangi variasi dalam teks. Kami melakukan tiga langkah dalam normalisasi teks, yaitu:
- _Lowercasing_: Mengubah semua karakter teks menjadi huruf kecil, sehingga tidak ada perbedaan antara huruf besar dan kecil dalam teks.
- _Remove Stop Words_: Menghapus kata-kata umum yang tidak memiliki makna penting dalam pemrosesan teks, seperti “dan”, “yang”, “di”, dll. Hal ini dapat mengurangi ukuran data dan waktu pemrosesan.
- _Lemmatisasi_: Mengubah kata-kata menjadi bentuk dasar (lemah) untuk mengurangi variasi kata yang sama, seperti “berjalan” menjadi “jalan”, “menyanyi” menjadi “nyanyi”, dll. Hal ini dapat meningkatkan kinerja model analisis teks dengan memfokuskan pada kata-kata yang memiliki makna atau informasi penting.

2. Pembersihan Teks: Teknik ini bertujuan untuk menyederhanakan teks dan menghilangkan informasi yang tidak relevan untuk analisis emosi. Kami melakukan tiga langkah dalam pembersihan teks, yaitu:
- Menghapus Angka: Menghilangkan angka dari teks, karena angka sering kali tidak memberikan kontribusi signifikan terhadap analisis sentimen atau emosi. Misalnya, angka “5” atau “10” tidak menunjukkan emosi apa pun dalam teks.
- Menghapus Tanda Baca: Menghilangkan tanda baca dari teks, karena tanda baca tidak selalu penting dalam analisis emosi. Misalnya, tanda koma (,) atau titik (.) tidak menunjukkan emosi apa pun dalam teks.
- Menghapus URL: Menghapus URL dari teks, karena URL tidak berkontribusi pada analisis emosi. Misalnya, URL “https://www.tokopedia.com/” tidak menunjukkan emosi apa pun dalam teks.

3. _Tokenisasi_: Teknik ini bertujuan untuk memisahkan teks menjadi urutan token atau kata-kata, sehingga dapat diolah lebih lanjut, seperti dalam proses pembuatan model. Kami menggunakan fungsi tokenizer yang disediakan oleh library keras untuk melakukan tokenisasi teks.

4. _Padding_: Teknik ini bertujuan untuk membuat semua teks memiliki panjang yang sama, sehingga dapat dimasukkan ke dalam model yang membutuhkan input dengan panjang yang sama. Kami menggunakan fungsi pad_sequences yang disediakan oleh library keras untuk melakukan padding teks.

### Sebelum dan sesudah penggunaan cleaning pada teks

Untuk memberikan gambaran tentang hasil dari teknik data preparation yang kami lakukan, kami menampilkan beberapa contoh teks sebelum dan sesudah penggunaan cleaning pada tabel berikut:

Table 7. Hasil Cleaning Customer Review:
| Before Cleaning                                               | After Cleaning                                         |
| ------------------------------------------------------------- | ------------------------------------------------------ |
| Alhamdulillah berfungsi dengan baik. Packaging...             | alhamdulillah berfungsi dengan baik packaging ...      |
| barang bagus dan respon cepat, harga bersaing ...             | barang bagus dan respon cepat harga bersaing d...      |
| barang bagus, berfungsi dengan baik, seler ram...             | barang bagus berfungsi dengan baik seler ramah...      |

## Modeling
***
### Tahapan dan Parameter Pemodelan
1. _TF-IDF Embedding_: 
- Digunakan untuk mengubah teks ke dalam representasi vektor numerik.
- Parameter: Beberapa parameter yang dapat disesuaikan dalam _TF-IDF_, seperti penggunaan _stopwords_, _n-grams_, dll.
- Alasan penggunaan: _TF-IDF_ dipilih karena merupakan metode yang umum digunakan dalam analisis teks dan dapat membantu mengurangi dimensi fitur yang besar.

2. _Logistic Regression_:
- _Logistic Regression_ adalah sebuah metode analisis statistik yang membangun sebuah model statistik untuk menggambarkan hubungan antara sebuah variabel hasil yang bersifat biner atau dikotomis (ya/tidak) dengan satu atau lebih variabel prediktor atau penjelas. Model ini menggunakan fungsi logit untuk memodelkan peluang dari sebuah kejadian biner[7](https://link.springer.com/referenceworkentry/10.1007/978-3-319-69909-7_1689-2).
- Parameter: Solver ('liblinear'), random_state.
- Kelebihan: Mudah diimplementasikan, memiliki interpretasi yang baik, efisien untuk dataset besar.
- Kekurangan: Tidak dapat menangani hubungan yang kompleks.
- Alasan penggunaan: Dipilih karena kemudahannya dalam implementasi, memiliki interpretasi yang baik, dan efisien untuk dataset besar. Namun, model ini tidak dapat menangani hubungan yang kompleks, sehingga cocok digunakan untuk kasus dengan hubungan yang relatif sederhana.

3. _Support Vector Machine_:
- Support Vector Machine (_SVM_) adalah sebuah metode pembelajaran terbimbing yang dapat digunakan untuk klasifikasi, regresi, estimasi densitas, deteksi anomali, dan aplikasi lainnya. Dalam kasus klasifikasi dua kelas yang paling sederhana, _SVM_ mencari sebuah bidang hiper yang memisahkan dua kelas data dengan margin sebesar-besarnya. Hal ini menghasilkan akurasi generalisasi yang baik pada data yang belum dilihat, dan mendukung metode optimisasi khusus yang memungkinkan _SVM_ untuk belajar dari jumlah data yang besar[8](https://link.springer.com/referenceworkentry/10.1007/978-0-387-30164-8_804).
- Parameter: Kernel (default='rbf'), random_state.
- Kelebihan: Efektif dalam ruang fitur dengan banyak dimensi, dapat menangani data non-linier.
- Kekurangan: Membutuhkan waktu komputasi yang lama untuk dataset besar.
- Alasan penggunaan: Dipilih karena efektif dalam ruang fitur dengan banyak dimensi dan mampu menangani data non-linier. Namun, _SVM_ membutuhkan waktu komputasi yang lama untuk dataset besar.

4. _LSTM_:
- _LSTM_ adalah singkatan dari Long Short-Term Memory, yaitu sebuah arsitektur jaringan saraf berulang (RNN) yang dirancang untuk mengatasi masalah gradien yang menghilang dan meledak pada RNN konvensional. Berbeda dengan jaringan saraf maju, RNN memiliki koneksi siklik yang membuatnya kuat untuk memodelkan urutan[9](https://link.springer.com/article/10.1007/s10462-020-09838-1).
- Parameter: Jumlah unit _LSTM_, dropout rate, batch size, jumlah epoch.
- Kelebihan: Efektif untuk memahami konteks dalam teks panjang, dapat menangani data sekuen.
- Kekurangan: Membutuhkan waktu dan sumber daya komputasi yang besar, rentan terhadap overfitting.
- Alasan penggunaan: Dipilih karena kemampuannya dalam memahami konteks dalam teks panjang dan menangani data sekuen. _LSTM_ cocok digunakan untuk dataset dengan data sekuensial seperti review pelanggan. Namun, model ini membutuhkan waktu dan sumber daya komputasi yang besar serta rentan terhadap overfitting.

### Proses Improvement dengan Hyperparameter Tuning
Pada model _LSTM_, melakukan improvement dengan melakukan hyperparameter tuning. Beberapa langkah yang dilakukan adalah:

1. Penambahan layer _LSTM_ dan penyesuaian jumlah unit _LSTM_.
2. Penyesuaian dropout rate untuk mengurangi overfitting.
3. Penggunaan BatchNormalization untuk mempercepat konvergensi model.
4. Penggunaan early stopping dan learning rate reduction untuk menghindari _overfitting_.

Proses improvement ini dilakukan untuk meningkatkan performa model _LSTM_ dalam menganalisis sentimen emosi pada data review pelanggan.

<br>

<div><img src="https://i.ibb.co/HhRvybW/download.png" width="300"/></div>

Gambar 8. Arsitektur _LSTM_ yang digunakan

<br>

### Pemilihan Model Terbaik
Dari ketiga model yang digunakan, model _LSTM_ dipilih sebagai model terbaik karena mampu memberikan kinerja yang lebih baik dalam memahami konteks teks panjang dan menangani data sekuen dengan baik. _LSTM_ memiliki kelebihan dalam menangani data yang bersifat sekuensial, seperti review pelanggan, dan dapat mengatasi masalah vanishing gradient yang sering terjadi pada model RNN tradisional.

## Evaluasi
***
Pada proyek ini menggunakan model deep learning bertipe klasifikasi yang berarti jika mendekati 100% accuracy, performanya bagus, sedangkan jika dibawah 75%, maka performanya jelek. Metrik yang akan pakai pada prediksi ini adalah Accuracy, metrik ini menghitung persentase prediksi yang benar dari total prediksi yang dilakukan. Accuracy ditulis dalam rumus berikut:

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
+ Table 8. Akurasi pada 3 model:
  | model               | accuracy |
  |---------------------|----------|
  | LSTM                | 0.98     |
  | SVM                 | 0.97     |
  | Logistic Regression | 0.82     |

    <div><img src="https://i.ibb.co/RTDt9L4/download-1.png" width="300"/></div>
    Gambar 9. Gambar 6. Kurva perbandingan akurasi model yang digunakan

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma _LSTM_ memiliki akurasi lebih tinggi tinggi dibandingkan algoritma lainnya dalam proyek ini.

+ Presisi, Recall, F1-Score
- _Logistic Regression_

Table 9. Classification Report _Logistic Regression_

|         | precision | recall | f1-score | support |
|---------|-----------|--------|----------|---------|
| Anger   | 0.90      | 0.67   | 0.77     | 675     |
| Fear    | 0.81      | 0.76   | 0.78     | 892     |
| Happy   | 0.81      | 0.95   | 0.87     | 1753    |
| Love    | 0.87      | 0.62   | 0.72     | 800     |
| Sadness | 0.79      | 0.90   | 0.84     | 1185    |

- _SVM_

Table 10. Classification Report _SVM_

|         | precision | recall | f1-score | support |
|---------|-----------|--------|----------|---------|
| Anger   | 0.98      | 0.95   | 0.97     | 675     |
| Fear    | 0.98      | 0.97   | 0.98     | 892     |
| Happy   | 0.97      | 0.99   | 0.98     | 1753    |
| Love    | 0.98      | 0.94   | 0.96     | 800     |
| Sadness | 0.96      | 0.99   | 0.97     | 1185    |

- _LSTM_

Table 11. Classification Report _LSTM_

|   | precision | recall | f1-score | support |
|---|-----------|--------|----------|---------|
| 0 | 0.98      | 0.98   | 0.98     | 675     |
| 1 | 0.99      | 0.96   | 0.97     | 892     |
| 2 | 0.99      | 0.99   | 0.99     | 1753    |
| 3 | 0.99      | 0.97   | 0.98     | 800     |
| 4 | 0.96      | 0.99   | 0.98     | 1185    |

Dari hasil evaluasi di atas, dapat disimpulkan beberapa hal:

Proyek ini bertujuan untuk mengembangkan sistem analisis emosi pada teks dari media sosial untuk membantu perusahaan memahami persepsi publik terhadap merek atau produk mereka. Melalui analisis emosi, perusahaan dapat meningkatkan kualitas layanan, menyesuaikan strategi pemasaran, dan memberikan layanan yang lebih baik kepada pelanggan.

Dalam proyek ini, telah menggunakan tiga pendekatan klasifikasi yang berbeda, yaitu _Logistic Regression_, Support Vector Machine, dan _LSTM_, untuk mengklasifikasikan emosi dalam teks media sosial. Setelah melakukan data preparation dan modeling, sudah mengevaluasi model menggunakan metrik akurasi, presisi, recall, dan F1-score.

Dari hasil evaluasi, model _LSTM_ memberikan kinerja yang lebih baik dibandingkan dengan model lainnya, dengan akurasi sebesar 0.98 dan nilai presisi, recall, dan F1-score tertinggi untuk semua kelas emosi. Hal ini menunjukkan bahwa model _LSTM_ mampu memahami konteks teks panjang dan mengklasifikasikan emosi dengan baik pada data review pelanggan.

Dengan demikian, proyek ini dapat dianggap berhasil dalam mengembangkan sistem analisis emosi pada teks media sosial untuk membantu perusahaan memahami persepsi publik terhadap merek atau produk mereka. Model _LSTM_ dapat menjadi solusi yang efektif dalam memprediksi emosi berdasarkan teks yang tersebar di media sosial, sehingga dapat membantu perusahaan meningkatkan kualitas layanan dan strategi pemasaran mereka.


Referensi:  
  [1]    
  [Nandwani P, Verma R. A review on sentiment analysis and emotion detection from text. Soc Netw Anal Min. 2021;11(1):1-19. doi:10.1007/s13278-021-00776-6.](https://jurnal.umk.ac.id/index.php/simet/article/view/3487)

  [2]  
  [Cheung AKL. Encyclopedia of Quality of Life and Well-Being.; 2021.](https://link.springer.com/article/10.1007/s13278-021-00776-6)

  [3]  
  [8 Sentiment Analysis Real-World Use Cases. https://www.repustate.com/blog/sentiment-analysis-real-world-examples/](https://www.repustate.com/blog/sentiment-analysis-real-world-examples/)

  [4]  
  [Social Media Sentiment Analysis: Tools, Tips, and More. https://blog.hootsuite.com/social-media-sentiment-analysis-tools/](https://blog.hootsuite.com/social-media-sentiment-analysis-tools/)
  
  [5]  
  [Benrouba F, Boudour R. Emotional sentiment analysis of social media content for mental health safety. Soc Netw Anal Min. 2023;13(1):0-11. doi:10.1007/s13278-022-01000-9.](https://www.semanticscholar.org/paper/Emotional-sentiment-analysis-of-social-media-for-Benrouba-Boudour/89682c3a79487a7af9895eb3067c80fc77804515)

  [6]  
  [Chandra R, Krishna A. COVID-19 sentiment analysis via deep learning during the rise of novel cases. PLoS One. 2021;16(8 August):1-26. doi:10.1371/journal.pone.0255615.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255615)

  [7]  
  [Cheung AKL. Encyclopedia of Quality of Life and Well-Being.; 2021.](https://link.springer.com/referencework/10.1007/978-3-319-69909-7)

  [8]  
  [Sammut C, Webb GI. Front Matter. Encycl Mach Learn. Published online 2011.](https://link.springer.com/referencework/10.1007/978-0-387-30164-8)

  [9]
  [Van Houdt G, Mosquera C, Nápoles G. A review on the long short-term memory model. Artif Intell Rev. 2020;53(8):5929-5955. doi:10.1007/s10462-020-09838-1.](https://link.springer.com/article/10.1007/s10462-020-09838-1)
<<<<<<< HEAD
=======

>>>>>>> 74d4be6d7392542e8619766b52b546b22e885388
