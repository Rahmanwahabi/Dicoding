# MLT Proyek Kedua | Book Recommendation System

###### Disusun oleh : Rahman Wahabi Hasibuan

Ini adalah proyek kedua, membuat sistem rekomendasi untuk memenuhi submission Dicoding Kelas *machine learning* Terapan. 

Proyek ini membangun model *machine learning* yang dapat memberikan rekomendasi buku kepada pengguna.

## 1. Project Domain

### Latar belakang

Era informasi membuka gerbang bagi lautan pengetahuan yang tak terbatas. Di tengah kelimpahan informasi ini, tantangan baru muncul: menemukan jarum dalam tumpukan jerami, yaitu buku yang tepat untuk mengasah imajinasi, memperluas wawasan, dan mengugah jiwa.

<br>

<div><img src="https://i.imgur.com/Ai2MY1N.jpeg" width="1000"/></div>

Gambar 1. Pentingnya membaca buku

<br>

Di sinilah sistem rekomendasi buku hadir sebagai kompas yang menuntun para petualang literasi menuju harta karun pengetahuan yang sesuai dengan minat dan kebutuhan mereka.

Industri penerbitan dan perpustakaan online berlomba-lomba menghadirkan inovasi ini untuk memanjakan para pembaca. *machine learning* menjadi mesin ajaib di balik sistem rekomendasi, menganalisis data dan mempelajari pola untuk memahami keinginan setiap individu.

Riwayat bacaan, ulasan, dan interaksi pengguna dengan buku menjadi jejak digital yang dipelajari oleh model *machine learning*. Pengolahan bahasa alami membantu model memahami makna teks deskripsi buku, tema, dan keterkaitan antar buku.

Informasi ini digabungkan dengan data pengguna, seperti preferensi genre, tingkat kesulitan bacaan, dan riwayat bacaan, untuk menghasilkan rekomendasi yang personal dan relevan. Berdasarkan latar belakang peneliti sebelumnya penulis mengambil kesimpulan untuk metode klasifikasi yang digunakan, diantaranya _Content Based Filtering_ dan _Collaborative Filtering_.

### Manfaat bagi perusahaan atau industri tertentu:

- Pembaca:
  - Menemukan buku yang sesuai dengan minat dan kebutuhan.
  - Meningkatkan pengalaman membaca.
  - Memperluas wawasan dan pengetahuan.

- Penerbit:
  - Meningkatkan visibilitas buku.
  - Menjangkau pembaca yang tepat.
  - Meningkatkan penjualan.

- Perpustakaan online:
  - Meningkatkan pengalaman pengguna.
  - Mempertahankan pembaca.
  - Meningkatkan kepuasan pengguna.

Sistem rekomendasi buku adalah jembatan yang menghubungkan pembaca dengan buku yang ditakdirkan untuk mereka. Teknologi ini membuka jalan menuju dunia literasi yang personal dan menyenangkan.

## 2. Business Understanding

### Problem Statements
Dalam era informasi yang serba cepat dan penuh dengan pilihan, pembaca sering kali menghadapi kesulitan dalam menemukan buku yang tepat yang sesuai dengan minat dan kebutuhan mereka. Dengan begitu banyaknya buku yang tersedia, proses pencarian buku yang relevan bisa menjadi sangat memakan waktu dan terkadang mengecewakan.

### Goals
Tujuan utama dari sistem rekomendasi buku ini adalah untuk:
- Memudahkan pembaca dalam menemukan buku yang relevan dengan minat dan kebutuhan mereka.
- Meningkatkan efisiensi pencarian buku dengan menyediakan rekomendasi yang akurat.
- Membantu penerbit dan perpustakaan online dalam meningkatkan visibilitas buku dan kepuasan pengguna.

### Solution Approach
Untuk mencapai tujuan tersebut, kita akan menerapkan dua pendekatan solusi:

1. **Content-Based Filtering**: Pendekatan ini akan menggunakan deskripsi buku, tema, dan metadata lainnya untuk merekomendasikan buku yang memiliki konten serupa dengan yang telah disukai atau dinilai tinggi oleh pengguna. Ini memungkinkan personalisasi yang kuat karena rekomendasi didasarkan pada preferensi unik pengguna.

2. **Collaborative Filtering**: Pendekatan ini akan menganalisis pola dan preferensi dari banyak pengguna untuk mengidentifikasi kesamaan antara pengguna dan memberikan rekomendasi berdasarkan apa yang disukai oleh pengguna lain yang memiliki selera serupa. Ini membantu dalam menemukan buku yang mungkin belum ditemukan atau dipertimbangkan oleh pengguna.

Kedua metode ini akan dikombinasikan untuk menghasilkan sistem rekomendasi yang komprehensif, memanfaatkan kekuatan dari kedua pendekatan untuk memberikan rekomendasi yang akurat dan personal kepada pengguna.

### Studi kasus tentang bagaimana _Book Recommendation System_ telah memberikan nilai tambah bagi perusahaan:


1. _Research-paper recommender systems: a literature survey_ yang diterbitkan di International Journal on Digital Libraries. Artikel ini memberikan tinjauan literatur tentang sistem rekomendasi artikel penelitian, termasuk sistem rekomendasi buku, dengan membahas konsep dan pendekatan yang paling umum digunakan[1](https://link.springer.com/article/10.1007/s00799-015-0156-0).

2. _BOOK RECOMMENDATION SYSTEM USING MACHINE LEARNING_ yang diterbitkan di IJCRT. Makalah ini membahas tentang proses merekomendasikan buku kepada pengguna dari semua kelompok umur dengan menggunakan metodologi collaborative filtering[2](https://ijcrt.org/papers/IJCRT2205341.pdf).

3. _Hybrid Book Recommendation System_ yang diterbitkan di IRJET. Makalah ini membahas tentang sistem rekomendasi buku hibrid yang menggunakan algoritma seperti SVD, KNN, RBM, dan rekomendasi hibrid[3](https://www.irjet.net/archives/V6/i7/IRJET-V6I7379.pdf).

## 3. Data Understanding
Dataset yang digunakan dalam proyek ini adalah[Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) berisi data yang dapat digunakan untuk membangun sistem rekomendasi buku canggih. Dataset ini biasanya mencakup informasi seperti ratings yang diberikan oleh pengguna untuk buku tertentu, metadata buku seperti judul, penulis, dan tahun penerbitan, serta interaksi pengguna dengan buku-buku tersebut. Data ini dapat digunakan untuk menganalisis preferensi pengguna dan membuat rekomendasi yang dipersonalisasi berdasarkan pola rating dan perilaku pengguna.

Deskripsi Data:

Dataset yang digunakan terdiri dari tiga file CSV:

- books.csv: berisi informasi tentang buku, termasuk ISBN, judul, penulis, tahun publikasi, penerbit, dan URL gambar.
- ratings.csv: berisi informasi tentang penilaian buku oleh pengguna, termasuk ID pengguna, ISBN buku, dan rating.
- users.csv: berisi informasi tentang pengguna, termasuk ID pengguna, lokasi, dan usia.


Tabel 1. Ringkasan Data. 

| Dataset     | Jumlah Sampel | Fitur                                                       |
|-------------|---------------|-------------------------------------------------------------|
| books.csv   | 271.360       | ISBN, Judul, Penulis, Tahun Publikasi, Penerbit, URL Gambar |
| ratings.csv | 1.149.780     | ID Pengguna, ISBN, Rating                                   |
| users.csv   | 2.278.858     | ID Pengguna, Lokasi, Usia                                   |

### Pemeriksaan Data

- Missing Value: terdapat missing value pada kolom 'Year-Of-Publication' dan 'Image-URL-S', 'Image-URL-M', 'Image-URL-L' pada dataset books.csv.
- Duplikat: terdapat duplikat data pada ketiga dataset.
- Tipe Data: beberapa kolom memiliki tipe data yang tidak sesuai, seperti 'Year-Of-Publication' yang seharusnya bertipe integer.

### Pembersihan Data

- Menghapus missing value pada kolom 'Year-Of-Publication'.
- Menghapus duplikat data pada ketiga dataset.
- Mengubah tipe data kolom 'Year-Of-Publication' menjadi integer.

### Analisis Data: Univariate Exploratory Data Analysis dan Multvarite Exploratory Data Analysis

1. Distribusi Penulis:

- Menghitung jumlah buku yang ditulis oleh setiap penulis.
- Menampilkan 10 penulis dengan jumlah buku terbanyak.

<br>

<div><img src="https://i.imgur.com/Ci9Kmgs.png" width="1000"/></div>

Gambar 2. Top 10 Authors by Number of Books

<br>

2. Distribusi Rating:

- Menghitung rata-rata rating untuk setiap buku.
- Menampilkan distribusi rating.

<br>

<div><img src="https://i.imgur.com/GPWTPcT.png" width="1000"/></div>

Gambar 3. Distribusi Rating

<br>

3. Korelasi antara Rating dan Tahun Publikasi:

- Menghitung rata-rata rating untuk setiap tahun publikasi.
- Memvisualisasikan hubungan antara rating dan tahun publikasi.

<br>

<div><img src="https://i.imgur.com/HU12cqp.png" width="1000"/></div>

Gambar 4. Plot rata-rata rating vs tahun

<br>

4. Korelasi antara Rating dan Jumlah Rating:

Menghitung jumlah rating untuk setiap buku.
Menghitung rata-rata rating untuk setiap buku.
Memvisualisasikan hubungan antara rating dan jumlah rating.

<br>

<div><img src="https://i.imgur.com/808KJ3n.png" width="1000"/></div>

Gambar 5. Plot Rata-rata rating vs Jumlah rating per buku

<br>

## 4. Data Preparation

### Persiapan Data untuk Model Development dengan _Content-Based Filtering_

1. Pembersihan Data

Langkah pertama adalah membersihkan data dengan membuang nilai yang hilang.

<br>

<div><img src="https://i.imgur.com/Rc9zZQo.jpeg" width="1000"/></div>

Gambar 6. DataFrame setelah *all_books_clean = books.dropna()*

<br>

Kemudian, periksa kembali apakah ada nilai yang hilang:

Tabel 2. Jumlah nilai yang hilang (NaN) di setiap kolom dalam DataFrame all_books_clean.

| ISBN                | 0 |
|---------------------|---|
| Book-Title          | 0 |
| Book-Author         | 0 |
| Year-Of-Publication | 0 |
| Publisher           | 0 |
| dtype: int64            |


2. Standarisasi Jenis Buku Berdasarkan ISBN

Buku-buku diurutkan berdasarkan ISBN dan disimpan dalam variabel fix_books:

<br>

<div><img src="https://i.imgur.com/QletnDO.png" width="1000"/></div>

Gambar 7. Urutan buku berdasarkan ISBN

<br>

3. Menghitung Jumlah Judul Buku Unik

Jumlah judul buku unik dihitung untuk memastikan proses deduplikasi diperlukan: 271354

4. Menghilangkan Duplikasi Data

Data duplikat di kolom ISBN dihapus dan disimpan dalam variabel preparation:
Membuat DataFrame baru bernama preparation dari DataFrame fix_books dengan menghapus baris duplikat berdasarkan kolom ‘ISBN’. Fungsi drop_duplicates('ISBN') memeriksa kolom ‘ISBN’ dan hanya menyimpan baris pertama untuk setiap nilai ISBN yang unik, sementara semua baris lain yang memiliki nilai ISBN yang sama akan dihapus. Ini membantu dalam memastikan bahwa setiap buku diwakili sekali saja dalam dataset, berdasarkan nomor ISBN mereka yang unik, sehingga menghindari redundansi data.

5. Mengubah Data Menjadi Daftar dan Memeriksa Panjang Daftar

Kolom data diubah menjadi daftar untuk memudahkan pemrosesan dan Panjang setiap daftar diperiksa untuk memastikan konsistensi:

| 271354       |
|--------------|
| 271354       |
| 271354       |
| 271354       |
| 271354       |

6. Membuat DataFrame Baru

Dictionary dibuat untuk menyimpan data dan kemudian diubah menjadi DataFrame baru bernama books_new:

<br>

<div><img src="https://i.imgur.com/wPMTkp4.png" width="1000"/></div>

Gambar 8. DataFrame baru bernama books_new

<br>

### Persiapan Data untuk Model Development dengan _Collaborative Filtering_:

1. Pengkodean Fitur:

Fitur User-ID dan ISBN diubah menjadi indeks bilangan bulat untuk meningkatkan efisiensi komputasi.

2. Memetakan Fitur ke DataFrame:

Fitur User-ID dan ISBN yang diubah dipetakan ke DataFrame untuk menghubungkan data dengan user dan buku.

3. Memeriksa Data:

Jumlah pengguna, judul buku, nilai minimum dan maksimum peringkat diperiksa untuk memastikan data siap digunakan.

## 5. Modeling

### Content-Based Filtering Model Development

1. Data yang digunakan:

- data berisi sebagian data dari books_new (dibatasi 20.000 baris)

2. TF-IDF Vectorizer:

- Digunakan untuk menangkap hubungan antar kata dalam deskripsi buku (biasanya kolom book_author).
- tf menyimpan vectorizer yang sudah difit dengan data book_author.
- tfidf_matrix menyimpan representasi berupa matriks TF-IDF dari data book_author.
3. Cosine Similarity:

- Digunakan untuk mengukur kemiripan antar dokumen (deskripsi buku) berdasarkan matriks TF-IDF.
- cosine_sim menyimpan nilai kemiripan antar dokumen.
- cosine_sim_df berupa DataFrame yang memudahkan visualisasi kemiripan antar judul buku.
4. Mendapatkan Rekomendasi:

Fungsi book_recommendation menerima judul buku dan mengembalikan rekomendasi buku lain yang mirip.

### Collaborative Filtering Model Development

1. Memisahkan Data untuk Training dan Validation:

- Data df_rating diacak terlebih dahulu.
- x_train dan y_train berisi data untuk training model.
- x_val dan y_val berisi data untuk validasi model.

2. Proses Training:

- Kelas RecommenderNet membangun model rekomendasi dengan network saraf.
- Model dilatih dengan data x_train dan y_train.
- Fungsi loss dan metrik evaluasi digunakan untuk mengukur performa model.

3. Mendapatkan Rekomendasi Judul Buku:

- book_df berisi keseluruhan data buku.
- ID pengguna (user_id) diambil secara acak.
- book_readed_by_user berisi buku yang pernah dibaca oleh pengguna tersebut.
- book_not_readed berisi buku yang belum dibaca oleh pengguna.
- user_book_array berisi kombinasi user ID dan buku yang belum dibaca.
- ratings_model memprediksi rating yang mungkin diberikan pengguna terhadap buku yang belum dibaca.
- Buku dengan rating prediksi tertinggi menjadi rekomendasi untuk pengguna.

## 6. Evaluation

### Content-Based Filtering
Metrik Evaluasi:

- Precision: Mengukur proporsi rekomendasi yang relevan dengan ground truth.
- Recall: Mengukur proporsi ground truth yang berhasil direkomendasikan.
- F1-score: Menggabungkan precision dan recall untuk memberikan gambaran menyeluruh tentang performa model.

Hasil Evaluasi:

<br>

<div><img src="https://i.imgur.com/CjHoPb1.png" width="1000"/></div>

Gambar 9. Hasil Evaluasi Content-Based Filtering

<br>


Nilai-nilai ini menunjukkan bahwa model rekomendasi berhasil dengan sempurna dalam merekomendasikan buku yang disukai pengguna.

Penjelasan Metrik:

- Precision:
precision = TP / (TP + FP)
- Recall:
recall = TP / (TP + FN)
- F1-score:
f1 = 2 * (precision * recall) / (precision + recall)

- TP: True Positive (rekomendasi yang relevan)
- FP: False Positive (rekomendasi yang tidak relevan)
- FN: False Negative (ground truth yang tidak direkomendasikan)


### Collaborative Filtering

Metrik Evaluasi:

Root Mean Squared Error (RMSE): Mengukur rata-rata kesalahan prediksi rating buku.

Hasil Evaluasi:

<br>

<div><img src="https://i.imgur.com/CjHoPb1.png" width="1000"/></div>

Gambar 10. Hasil Evaluasi Collaborative Filtering

<br>

Hasil evaluasi menunjukkan bahwa model memiliki kinerja yang sangat baik selama pelatihan. Grafik yang disajikan menampilkan root mean squared error (RMSE) selama berbagai epoch untuk dataset pelatihan dan pengujian:

- Training: Error pada data latihan menunjukkan penurunan yang konsisten seiring bertambahnya epoch, yang mengindikasikan bahwa model belajar dengan baik dari data.
- Test: Error pada data uji menurun dengan cepat pada awalnya, namun kemudian stabil. Ini bisa menandakan bahwa model telah mencapai titik di mana pembelajaran tambahan dari data pelatihan tidak lagi memberikan peningkatan signifikan pada kinerja terhadap data uji.


Penjelasan Metrik:

rmse = sqrt(mean((y_true - y_pred)^2))

- y_true: Rating buku yang sebenarnya
- y_pred: Rating buku yang diprediksi oleh model


### Kesimpulan
Kedua model menunjukkan performa yang cukup baik dalam memberikan rekomendasi. Content-based filtering memiliki precision yang tinggi, menunjukkan bahwa rekomendasinya relevan dengan preferensi pengguna. Collaborative filtering memiliki RMSE yang kecil, menunjukkan bahwa rating yang diprediksi oleh model cukup akurat.

Pemilihan model terbaik tergantung pada kebutuhan dan preferensi pengguna. Content-based filtering lebih cocok untuk pengguna yang ingin menemukan buku baru yang mirip dengan buku yang mereka sukai. Collaborative filtering lebih cocok untuk pengguna yang ingin menemukan buku yang sesuai dengan rating pengguna lain.


Referensi:  
  [1]    
  [Beel, J., Gipp, B., Langer, S., & Breitinger, C. (2016). Research-paper recommender systems: a literature survey. International Journal on Digital Libraries, 17(4), 305–338. https://doi.org/10.1007/s00799-015-0156-0.](ttps://link.springer.com/article/10.1007/s00799-015-0156-0)

  [2]  
  [PRASAD, D. M. (2023). Book Recommendation System Using Python. Interantional Journal of Scientific Research in Engineering and Management, 07(07), 39–43. https://doi.org/10.55041/ijsrem24710](https://ijcrt.org/papers/IJCRT2205341.pdf)

  [3]  
  [Vaidya, A., & Shinde, S. (2019). Hybrid Book Recommendation System. International Research Journal of Engineering and Technology, 3569(July), 3569–3577. www.irjet.net](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)


