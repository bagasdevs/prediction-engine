# Data Insert Tools

Folder ini berisi tools untuk mengelola data sensor database secara terpisah dari sistem utama untuk menghindari konflik database connection.

## 📁 Files

- `data_inserter.py` - Tool utama untuk insert data sensor
- `setup_initial_data.py` - Setup data awal untuk database
- `README.md` - Dokumentasi ini

## 🚀 Quick Start

### 1. Setup Data Awal (Wajib dilakukan sekali)
```bash
cd data_insert
python setup_initial_data.py
```

### 2. Insert Data Manual

#### Insert single record:
```bash
python data_inserter.py single
```

#### Insert batch records:
```bash
python data_inserter.py batch 50
```

#### Insert continuous (untuk testing):
```bash
python data_inserter.py continuous 5 10
```
*Insert setiap 5 detik selama 10 menit*

#### Check current count:
```bash
python data_inserter.py count
```

### 3. Interactive Mode
```bash
python data_inserter.py
```

## ⚠️ PENTING

- **JANGAN jalankan bersamaan dengan sistem utama**
- Tools ini dirancang untuk dijalankan terpisah
- Hentikan sistem utama dulu sebelum menggunakan tools ini
- Setelah insert data, baru jalankan sistem utama

## 🔄 Workflow Recommended

1. **Setup awal:**
   ```bash
   cd data_insert
   python setup_initial_data.py
   ```

2. **Jalankan sistem utama:**
   ```bash
   cd ..
   python start_system.py
   ```

3. **Jika perlu tambah data (hentikan sistem dulu):**
   ```bash
   # Stop sistem utama dengan Ctrl+C
   cd data_insert
   python data_inserter.py batch 100
   cd ..
   python start_system.py  # Restart sistem
   ```

## 📊 Data Generated

Tools ini menghasilkan data sensor realistis:
- **pH**: 5.5 - 9.0 (normal range 6.5-8.0)
- **Temperature**: 18-35°C (normal range 20-30°C)
- **Quality**: 'baik', 'sedang', 'buruk' (berdasarkan pH dan suhu)

## 🎯 Target Data

- **Minimum untuk AI**: 500 records
- **Target optimal**: 1000+ records
- **Historical data**: Data dengan timestamp bervariasi untuk training

## 💡 Tips

- Gunakan `batch` mode untuk insert data banyak dengan cepat
- Gunakan `continuous` mode untuk simulate real-time data
- Check `count` secara berkala untuk monitor progress
- Setup awal sudah include historical data untuk better AI training
