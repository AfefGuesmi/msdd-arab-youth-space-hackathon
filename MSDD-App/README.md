# MSDD - Marine Surface Debris Detection App

MSDD is a Flutter app that shows where marine debris was detected in satellite imagery. The detections are read from a CSV report, grouped into zones, and drawn on an interactive map. Each zone has a danger level, a recommended action and a qualitative-result image, and can be exported as a PDF report.

## Features

- **Interactive map** with a dark basemap. Zone icons appear when you zoom in and follow the area you move to.
- **Danger colours**: red = alert (high), orange = warning (medium), yellow = notice (low).
- **Detected zones sidebar**: every zone in a list. Selecting one zooms the map to it and shows its details and image. Tapping the zone on the map opens a popup.
- **Reports sidebar**: download a PDF for one zone, or a global PDF for all zones (details table, with the image under it).
- **Notifications sidebar**: one notification per detected area (Alert / Warning / Notice). Tapping one jumps to that area.
- **Accounts**: sign up and sign in, stored locally in SQLite (salted, hashed passwords). Sign up goes straight to the map; sign out is in the menu.
- **Assistant** icon in the menu ("Ask MSDD Assistant"). It has no page yet.
- **About the app** popup on the home page.

## Technologies

| Area                  | Technology                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| Framework / language  | Flutter, Dart                                                                                              |
| Map                   | [`flutter_map`](https://pub.dev/packages/flutter_map) with [`latlong2`](https://pub.dev/packages/latlong2) |
| Map tiles             | Esri ArcGIS "Canvas / World Dark Gray" (free, no API key)                                                  |
| Local database        | SQLite through [`sqflite`](https://pub.dev/packages/sqflite) (+ `path`)                                    |
| Password hashing      | [`crypto`](https://pub.dev/packages/crypto): SHA-256 with a per-account salt, 10,000 rounds                |
| PDF reports           | [`pdf`](https://pub.dev/packages/pdf) and [`printing`](https://pub.dev/packages/printing)                  |
| Phone notifications   | [`flutter_local_notifications`](https://pub.dev/packages/flutter_local_notifications) (Android)            |
| SQLite in the browser | [`sqflite_common_ffi_web`](https://pub.dev/packages/sqflite_common_ffi_web) (web / Docker only)            |
| Data                  | CSV report, UTM → latitude/longitude conversion written in Dart                                            |
| Container             | Docker, multi-stage build, nginx                                                                           |

Exact package versions are in `pubspec.yaml`.

## Requirements

**To run without Docker**

- [Flutter SDK](https://docs.flutter.dev/get-started/install), a recent stable release (3.27 or newer recommended; the Dart version required is in `pubspec.yaml` under `environment: sdk`).
- For Android: Android Studio (Android SDK + a JDK) and an emulator or a phone with USB debugging.
- For the browser: Google Chrome.
- An internet connection (the map tiles are downloaded online).

**To run with Docker**

- Docker 20.10 or newer (Docker Desktop on Windows/macOS, or Docker Engine on Linux). BuildKit is on by default in current versions.
- An internet connection for the first build (downloads the Flutter image, which is large, and the packages).
- Free port `8080` (changeable).
- Several GB of free disk space for the build image.


## Run without Docker

1. **Get the code** and open a terminal in the project folder (the one containing `pubspec.yaml`).

2. **Declare the assets** in `pubspec.yaml`. All three folders must be listed:

   ```yaml
   flutter:
     assets:
       - assets/images/
       - assets/qualitative_results/
       - assets/files/
   ```

3. **Install the packages**:

   ```bash
   flutter pub get
   ```

4. **Check your setup** (optional): `flutter doctor`.

5. **Run the app**:

   ```bash
   flutter devices            # list emulators / phones / Chrome
   flutter run                # Android emulator or connected phone
   flutter run -d chrome      # in the browser (see "Web notes" below)
   ```

   Browser note: the app uses SQLite for accounts, so for the browser follow "Web notes" first.

6. **Build a release version** (optional):

   ```bash
   flutter build apk --release        # Android APK  → build/app/outputs/flutter-apk/
   flutter build web --release        # web files    → build/web/
   ```
   
## Run with Docker

The Docker image builds the **web** version of the app and serves it with nginx. (Docker cannot run the Android/iOS version.)

### One-time preparation (browser support for accounts)

SQLite does not exist natively in a browser, so the web version uses `sqflite_common_ffi_web`. This is already prepared in the files delivered with this README:

1. Add the package:

   ```bash
   flutter pub add sqflite_common_ffi_web
   ```

2. Use the provided `lib/main.dart` and `lib/components/map.dart`. In `main.dart`, a browser run switches `sqflite` to its web version; in `map.dart`, phone notifications are skipped in a browser. Android behaviour is unchanged.

3. Make sure the project has a `web/` folder. If not, create it once and keep it:

   ```bash
   flutter create . --platforms=web
   ```

   (The Dockerfile also tries this automatically if `web/` is missing.)

4. Make sure the assets are declared in `pubspec.yaml` (step 2 of "Run without Docker").

### Build and run

```bash
docker build -t msdd .
docker run --rm -p 8080:80 msdd
```

Open <http://localhost:8080>.

Or with Docker Compose:

```bash
docker compose up --build
```

Useful options:

```bash
# Use the same Flutter version as on your machine (see `flutter --version`)
docker build --build-arg FLUTTER_IMAGE=ghcr.io/cirruslabs/flutter:3.35.0 -t msdd .

# Another port, e.g. 3000
docker run --rm -p 3000:80 msdd

# Run in the background / stop it
docker run -d --name msdd -p 8080:80 msdd
docker stop msdd && docker rm msdd
```

(Optional) build the Android APK with Docker (builds it, does not run it):

```bash
docker run --rm -v "$PWD":/app -w /app ghcr.io/cirruslabs/flutter:stable flutter build apk --release
```
