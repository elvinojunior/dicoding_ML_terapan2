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

* Bagaimana cara merekomendasikan film Indonesia berdasarkan kemiripan konten (judul, deskripsi, dan genre)?
* Seberapa baik performa sistem rekomendasi dalam menghasilkan rekomendasi yang relevan berdasarkan metrik evaluasi standar?

### Goals

* Membangun sistem rekomendasi film Indonesia berbasis **content-based filtering**.
* Mengevaluasi performa sistem menggunakan metrik **Precision@5**.

### Solution Statements

* Menggunakan **TF-IDF Vectorizer** pada kolom gabungan `title`, `description`, dan `genre`.
* Membangun fungsi rekomendasi yang dapat menampilkan top-N film teratas berdasarkan **cosine similarity**.

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

* Genre film terbanyak pada data
  ![image](https://github.com/user-attachments/assets/b62fa147-a989-4a77-bf1e-fa20703c4c9b)

* Distribusi user rating
  ![image](https://github.com/user-attachments/assets/a435fa25-76a8-40b6-88d2-1836a4aa000f)

* Kategori rating film pada data
  ![image](https://github.com/user-attachments/assets/865c4fd4-d46d-49d9-8ad8-b12a140449bc)

### Multivariate EDA
* Film populer pada masing masing genre  
  ![image](https://github.com/user-attachments/assets/9d0473d4-c8c5-48b4-bc38-951bb00a6643)

* Genre dengan user rating tertinggi  
  ![image](https://github.com/user-attachments/assets/429ced60-923a-4330-a0b7-01b4b8667855)

* Heatmap fitur numerik  
  ![image](https://github.com/user-attachments/assets/e60aab4d-4ab9-4325-bb60-6a28c67fc07a)

---
## Data Preparation

### Tahapan:
* Membuat kolom `combined` berisi gabungan:

  ```python
  df['combined'] = df['title'] + ' ' + df['description'] + ' ' + df['genre']
  ```
* Melakukan **TF-IDF vectorization** pada kolom `combined` menggunakan **stop words bahasa Inggris**.
* Melakukan **normalisasi hasil TF-IDF** agar vektor memiliki panjang seragam sebelum dihitung similarity-nya.

---

## Model Development

### Content-Based Filtering

* Menggunakan **TF-IDF Vectorizer** untuk mengubah teks di `combined` menjadi vektor numerik.
* Menghitung **cosine similarity** antar film berdasarkan vektor TF-IDF berikut ini Formula nya :  
  ![image](https://github.com/user-attachments/assets/a39de87b-966d-4b42-be73-427cacaa8af8)  
  Keterangan:  
```
𝑠𝑖𝑚(𝐴, 𝐵) = nilai similaritas dari item A dan item B
𝑛(𝐴) = banyaknya fitur konten item A 
𝑛(𝐵) = banyaknya fitur konten item B 
𝑛(𝐴 ∩ 𝐵)  = banyaknya fitur konten yang terdapat pada item A dan juga terdapat pada item B
```


---

## Model Evaluation

### 📊 Precision\@5

Untuk mengukur kualitas rekomendasi, dilakukan evaluasi **Precision\@5**, yaitu proporsi film relevan dalam 5 rekomendasi teratas untuk sebuah film.

**Definisi relevan**:
Film dengan **users\_rating ≥ 6.0** dianggap relevan.

### Hasil Precision\@5

* **Average Precision\@5 = 0.48**  
  Artinya, rata-rata dalam 5 film rekomendasi teratas, sekitar **2-3 film** benar-benar relevan.

---

## Hasil Testing

Contoh rekomendasi untuk film **Dilan 1990**:

| Rekomendasi            | Skor Similarity |
| :--------------------- | :-------------- |
| Dilan 1991             | 0.4243          |
| Milea                  | 0.3980          |
| #FriendButMarried      | 0.0812          |
| Rindu Kami Padamu      | 0.0712          |
| From Bandung with Love | 0.0637          |

**Insight:**  
Rekomendasi teratas adalah film sekuel atau film dengan genre dan tema serupa, yang menunjukkan sistem berhasil mengenali kemiripan konten.

---

## 📌 Kesimpulan

1. **Content-Based Filtering dengan TF-IDF** efektif merekomendasikan film Indonesia berdasarkan kemiripan deskripsi, genre, dan judul.
2. **Precision\@5 sebesar 0.48** menunjukkan performa yang cukup baik untuk sistem rekomendasi berbasis konten.
3. Model mampu merekomendasikan film-film yang relevan secara konteks, terutama untuk film dengan genre sejenis.
4. Sistem dapat dikembangkan lebih lanjut dengan menambahkan fitur `actors`, `directors`, dan metadata lain di kolom `combined`.

---

## Catatan

📓 Seluruh proses EDA, data preparation, TF-IDF, cosine similarity, dan evaluasi Precision\@5 dapat dilihat langsung di notebook terlampir.

---
