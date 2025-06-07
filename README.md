# Laporan Proyek Machine Learning : Sistem Rekomendasi Film Indonesia — Elvino Junior

## Domain Proyek

**Domain:** Entertainment / Film Recommendation System  
**Judul:** Sistem Rekomendasi Film Indonesia Menggunakan Content-Based Filtering


![image](https://github.com/user-attachments/assets/3f28a4e1-60a6-468a-90a7-cda752f91c2f)

## Latar Belakang

Di era digital saat ini, industri perfilman Indonesia mengalami perkembangan pesat dengan ratusan judul film diproduksi setiap tahunnya. Namun, penonton sering kali mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka di antara banyaknya pilihan yang tersedia. Fenomena ini dikenal sebagai "information overload" dalam dunia hiburan digital.  

Sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan konten yang relevan dengan minat mereka. Menurut penelitian oleh Ricci et al. (2015), sistem rekomendasi yang efektif dapat meningkatkan kepuasan pengguna hingga 35% dan engagement platform hingga 60%. Dalam konteks perfilman Indonesia, sistem rekomendasi dapat membantu mempromosikan film-film lokal dan meningkatkan apresiasi masyarakat terhadap karya sinema Indonesia.  

Content-Based Filtering merupakan salah satu pendekatan yang efektif untuk sistem rekomendasi, terutama ketika data interaksi pengguna terbatas. Pendekatan ini memanfaatkan karakteristik konten (seperti genre, deskripsi, dan metadata lainnya) untuk mengidentifikasi kesamaan antar item. Penelitian oleh Pazzani & Billsus (2007) menunjukkan bahwa Content-Based Filtering dapat mencapai akurasi yang tinggi dalam domain perfilman, khususnya ketika dikombinasikan dengan teknik Natural Language Processing.  

Proyek ini penting untuk dikembangkan karena dapat membantu platform streaming lokal, bioskop digital, atau aplikasi entertainment Indonesia dalam memberikan rekomendasi film yang lebih personal dan akurat kepada pengguna mereka.  

---

## Business Understanding

### Problem Statements

* Bagaimana cara merekomendasikan film Indonesia berdasarkan kemiripan konten?
* Seberapa baik performa sistem rekomendasi dalam menghasilkan rekomendasi yang relevan berdasarkan metrik evaluasi standar?

### Goals

* Membangun sistem rekomendasi film Indonesia berbasis **content-based filtering**.
* Mengevaluasi performa sistem menggunakan metrik **Precision@5**.

### Solution Statements

* Menggunakan **TF-IDF Vectorizer** pada kolom gabungan `title`, `description`, dan `genre`.
* Membangun fungsi rekomendasi yang dapat menampilkan top-5 film teratas berdasarkan **cosine similarity**.

---

## Data Understanding

![image](https://github.com/user-attachments/assets/1474d008-5326-4704-bfc3-b2a2786fd397)

Dataset yang digunakan berasal dari [Kaggle-IMDB Indonesian Movies](https://www.kaggle.com/datasets/dionisiusdh/imdb-indonesian-movies)

Struktur Data :

* **Jumlah data:** 1272 film
* **Jumlah fitur:** 11 kolom

**Deskripsi Variabel:**

![image](https://github.com/user-attachments/assets/ce993b82-61a3-486f-a6ad-60654cc8cbf6)

* `title` : Judul dari film       
* `year` : Tahun rilis film          
* `description` : Deskripsi film   
* `genre` : Genre film        
* `rating` : kategori rating film       
* `users_rating` : rating dari user  
* `votes` : Jumlah suara atau ulasan yang diberikan untuk film          
* `languages` : Bahasa dalam film      
* `directors` : Sutradara pada film     
* `actors` : Pemeran dalam film
* `runtime` : Durasi film     

---
## Data Cleaning
* Nilai kosong :
  + `description` : 432  
    Mengisi nilai kosong di kolom deskripsi dengan 'unknown'
  + `genre` : 36  
    Mengisi nilai kosong di kolom genre dengan 'unknown'
  + `rating` : 896  
    Mengisi nilai kosong di kolom rating dengan 'unrated' serta mengubah 11 kategori rating menjadi 5 kategori rating
  + `directors` : 7  
    Mengisi nilai kosong di kolom directors dengan 'unknown'
  + `runtime` :403  
    Mengisi nilai kosong di kolom runtime dengan 'unknown'
* Nilai duplikasi : Tidak ada nilai duplikasi  
* Mengubah tipe data pada kolom :
  + `votes` : mengubah tipe data object menjadi integer

---
## Exploration Data Analysis (EDA)
### Univariate EDA
* Distribusi film berdasarkan tahun  
  ![image](https://github.com/user-attachments/assets/f2c94fd1-05d5-41d0-b714-b8cc38078e9c)
  
Analisis ini bertujuan untuk mengetahui tren produksi film Indonesia dari tahun ke tahun. Dengan memplot jumlah film yang dirilis setiap tahun, dapat diamati apakah industri perfilman Indonesia mengalami pertumbuhan, stagnasi, atau penurunan. Hasil visualisasi menunjukkan peningkatan signifikan dalam jumlah film yang diproduksi setelah tahun-tahun tertentu, menandakan adanya perkembangan industri film lokal.

* Genre film terbanyak pada data  
  ![image](https://github.com/user-attachments/assets/b62fa147-a989-4a77-bf1e-fa20703c4c9b)
  
Analisis ini mengidentifikasi genre paling populer atau paling sering diproduksi di Indonesia. Dengan menghitung frekuensi tiap genre, diketahui genre mana yang mendominasi dataset. Hasilnya memperlihatkan beberapa genre seperti drama, komedi, dan horror menjadi genre favorit, yang dapat digunakan sebagai dasar dalam mengembangkan sistem rekomendasi agar lebih relevan.


* Distribusi user rating  
  ![image](https://github.com/user-attachments/assets/a435fa25-76a8-40b6-88d2-1836a4aa000f)
  
Distribusi ini menggambarkan bagaimana penilaian pengguna terhadap film-film Indonesia. Dengan melihat sebaran nilai `users_rating`, dapat diketahui bahwa kebanyakan film mendapat rating 6 sampai 7. Pola sebaran ini juga membantu dalam menentukan threshold relevansi pada tahap evaluasi model.

* Kategori rating film pada data  
  ![image](https://github.com/user-attachments/assets/865c4fd4-d46d-49d9-8ad8-b12a140449bc)
  
Kategori rating mengacu pada klasifikasi umur/konten (seperti SU, R, D, dsb). Analisis ini memeriksa bagaimana distribusi film berdasarkan kategori rating, yang penting untuk memastikan sistem rekomendasi tidak merekomendasikan film yang tidak sesuai dengan preferensi usia pengguna.

### Multivariate EDA
* Film populer pada masing masing genre  
  ![image](https://github.com/user-attachments/assets/9d0473d4-c8c5-48b4-bc38-951bb00a6643)
  
Analisis ini menghubungkan antara popularitas (user_rating) dengan genre film. Tujuannya untuk mengetahui genre apa yang memiliki film-film yang disukai user. Informasi ini bermanfaat untuk mengidentifikasi genre yang layak difokuskan pada sistem rekomendasi.

* Genre dengan user rating tertinggi  
  ![image](https://github.com/user-attachments/assets/429ced60-923a-4330-a0b7-01b4b8667855)

Analisis ini mencari genre yang rata-rata memiliki penilaian pengguna paling tinggi yaitu genre 'History'. Genre dengan rating rata-rata tinggi dapat dianggap memiliki kualitas konten yang baik


---
## Data Preparation

### Tahapan:
* Membuat kolom `combined` berisi gabungan:

  ```python
  df['combined'] = df['title'] + ' ' + df['description'] + ' ' + df['genre']
  ```
* Melakukan **TF-IDF vectorization** pada kolom `combined` menggunakan **stop words bahasa Inggris**.  
  ```python
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(df['combined'])
  ```
  
* Melakukan **normalisasi hasil TF-IDF** agar vektor memiliki panjang seragam sebelum dihitung similarity-nya.  
  ```python
  normalizer = Normalizer()
  tfidf_norm = normalizer.fit_transform(tfidf_matrix)
  ```

---

## Modeling and Result

### Model : Content-Based Filtering

* Menghitung **cosine similarity** antar film berdasarkan vektor TF-IDF berikut ini Formula nya :  
  ![image](https://github.com/user-attachments/assets/a39de87b-966d-4b42-be73-427cacaa8af8)  
    Keterangan:  
  ```
  𝑠𝑖𝑚(𝐴, 𝐵) = nilai similaritas dari item A dan item B
  𝑛(𝐴) = banyaknya fitur konten item A 
  𝑛(𝐵) = banyaknya fitur konten item B 
  𝑛(𝐴 ∩ 𝐵)  = banyaknya fitur konten yang terdapat pada item A dan juga terdapat pada item B
  ```

* Hasil dari perhitungan **cosine similarity** disimpan ke dalam dataframe untuk menilai kemiripan konten film
  ```python
  similarity_df = pd.DataFrame(similarity_cb, index=df['title'], columns=df['title'])
  ```
  
#### Kelebihan:
1. Tidak Memerlukan Data User  
  Sistem hanya butuh informasi dari item itu sendiri (judul, deskripsi, genre), tanpa perlu data interaksi pengguna.

2. Bisa Merekomendasikan Film Baru  
  Film yang belum pernah ditonton siapa pun tetap bisa direkomendasikan selama memiliki konten deskripsi/genre.

3. Personalisasi Berdasarkan Konten  
  Rekomendasi film dihasilkan berdasarkan kemiripan konten, sehingga hasilnya bisa sesuai minat user jika preferensinya diketahui.

4. Mudah Diimplementasikan  
  Prosesnya cukup dengan TF-IDF vectorization + cosine similarity, tanpa algoritma training yang berat.

5. Bebas dari Masalah Cold-Start User  
  Karena tidak butuh data aktivitas user, pengguna baru tetap bisa dapat rekomendasi bagus.

#### Kekurangan:
1. Fokus Terbatas ke Karakteristik Item  
  Hanya melihat kemiripan konten antar item tanpa mempertimbangkan perilaku atau rating pengguna lain, sehingga bisa melewatkan film bagus yang beda genre tapi disukai user serupa.

2. Rekomendasi Cenderung Monoton  
  Rekomendasi biasanya sangat mirip konten, sehingga user bisa bosan karena film yang direkomendasikan terlalu serupa.

3. Tidak Bisa Menangkap Tren Kolektif  
  Sistem tidak tahu film mana yang sedang populer atau disukai banyak orang, karena tidak mempertimbangkan data komunitas.

4. Fitur Konten Harus Lengkap dan Berkualitas  
  Jika deskripsi atau genre film banyak yang kosong atau tidak konsisten, performa rekomendasi bisa menurun.

5. Rentan Terhadap Skema Kata  
  Jika deskripsi film tidak konsisten atau terlalu pendek, cosine similarity antar TF-IDF vektornya bisa terlalu rendah meski sebetulnya filmnya mirip.

---

### Result

Contoh rekomendasi untuk film **Dilan 1990**:

| Rekomendasi            | Skor Similarity |
| :--------------------- | :-------------- |
| Dilan 1991             | 0.4243          |
| Milea                  | 0.3980          |
| #FriendButMarried      | 0.0812          |
| Rindu Kami Padamu      | 0.0712          |
| From Bandung with Love | 0.0637          |

**Insight:**  
Rekomendasi teratas adalah film sekuel dan film dengan genre dan deskripsi serupa, yang menunjukkan sistem berhasil mengenali kemiripan konten.

---
## Model Evaluation

### Visualisasi matriks similarity  
![image](https://github.com/user-attachments/assets/a55c4819-6d2b-42b0-ad69-5c714f76cbb9)  

Matriks similarity menunjukkan rendahnya kemiripan antar sebagian besar film berdasarkan konten teks TF-IDF (banyak nilai 0.00). Ini menyiratkan konten film yang sangat beragam/unik.

Namun, terdapat poin kemiripan menarik seperti "Aku tahu kapan kamu mati" dengan "4 Mantan" (0.04) yang menjadi kandidat utama untuk rekomendasi berbasis konten dan menunjukkan model berhasil menangkap koneksi khusus.


### Precision\@5  
![image](https://github.com/user-attachments/assets/75a67332-39d4-4870-af4f-9384a36d1322)

Untuk mengukur kualitas rekomendasi, dilakukan evaluasi **Precision\@5**, yaitu proporsi film relevan dalam 5 rekomendasi teratas untuk sebuah film.

**Definisi relevan**:
Film dengan **users\_rating ≥ 6.0** dianggap relevan.

### Hasil Precision\@5

* **Average Precision\@5 = 0.48**  
  Artinya, rata-rata dalam 5 film rekomendasi teratas, sekitar **2-3 film** benar-benar relevan.

---

## Kesimpulan

1. Business Problem pertama:  
  Sistem ini berhasil merekomendasikan film berdasarkan kemiripan konten tanpa membutuhkan data interaksi pengguna. Rekomendasi yang dihasilkan menunjukkan film-film dengan deskripsi, genre, dan judul yang berkaitan erat, seperti rekomendasi Dilan 1991 dan Milea untuk Dilan 1990, membuktikan efektivitas metode content-based dalam mengenali kemiripan isi.

2. Business Problem kedua:  
  Model berhasil mencapai Average Precision@5 sebesar 0.48, yang berarti rata-rata 2-3 dari 5 film rekomendasi teratas benar-benar relevan sesuai kriteria users_rating ≥ 6.0. Nilai precision ini cukup baik untuk baseline sistem rekomendasi berbasis konten, mengingat tanpa menggunakan data interaksi user (collaborative).

3. Insight tambahan dari EDA:  
  Ditemukan bahwa genre drama, komedi, dan horror mendominasi industri film Indonesia, dan genre history memiliki rata-rata user rating tertinggi. Temuan ini bisa menjadi bahan pertimbangan dalam pengembangan sistem rekomendasi ke depan agar lebih mempertimbangkan faktor genre.

Secara keseluruhan, sistem rekomendasi ini siap diimplementasikan pada platform film Indonesia berbasis streaming atau katalog digital untuk membantu pengguna menemukan film yang sesuai preferensi mereka secara lebih cepat dan relevan.

---
## Referensi 
1. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer. Diakses dari[https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_1]
2. Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. In The adaptive web (pp. 325-341). Springer. Diakses dari [https://link.springer.com/chapter/10.1007/978-3-540-72079-9_10]
3. Aggarwal, C. C. (2016). Recommender systems: the textbook. Springer. Diakses dari [https://link.springer.com/book/10.1007/978-3-319-29659-3]
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to information retrieval. Cambridge University Press. Diakses dari [https://www.cis.uni-muenchen.de/~hs/teach/14s/ir/pdf/19web.pdf]
5. Scikit-learn Documentation. (2023). TfidfVectorizer. Diakses dari [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html]
6. Badan Ekonomi Kreatif Indonesia. (2020). Data Statistik dan Hasil Survei Ekonomi Kreatif. Jakarta: BEKRAF. [https://kemenparekraf.go.id/publikasi-statistik-ekonomi-kreatif/statistik-ekonomi-kreatif-2020]
---
## Catatan

Seluruh kode implementasi, analisis EDA, preprocessing, model development, dan evaluasi tersedia dalam Jupyter notebook terlampir.

![image](https://github.com/user-attachments/assets/dc1efa08-ad05-455d-b3f2-166bc06faff9)


---
