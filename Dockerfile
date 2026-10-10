# syntax=docker/dockerfile:1
#
# MSDD — builds the Flutter WEB version of the app and serves it with nginx.
#
#   docker build -t msdd .
#   docker run --rm -p 8080:80 msdd        ->  http://localhost:8080
#
# Docker can only run the web build. The Android/iOS versions are built and
# run with the Flutter SDK on your own machine (see README.md).

# Image that contains the Flutter SDK. Use the same Flutter version you
# develop with, for example:
#   docker build --build-arg FLUTTER_IMAGE=ghcr.io/cirruslabs/flutter:3.35.0 -t msdd .
ARG FLUTTER_IMAGE=ghcr.io/cirruslabs/flutter:stable

# ---------------------------------------------------------------------------
# Stage 1 — build the web app
# ---------------------------------------------------------------------------
FROM ${FLUTTER_IMAGE} AS build

# Flutter refuses to run in a folder git thinks belongs to someone else.
RUN git config --global --add safe.directory '*'

WORKDIR /app

# Dependencies first, so this layer is cached until pubspec.yaml changes.
COPY pubspec.yaml pubspec.lock* ./
RUN flutter pub get

# The rest of the project (see .dockerignore for what is left out).
COPY . .

# Add the web platform if the project doesn't have a web/ folder yet.
# (Running "flutter create . --platforms=web" once on your machine and keeping
# the web/ folder is cleaner — this is only a fallback.)
ARG APP_NAME=msdd
RUN [ -d web ] || flutter create . --platforms=web --project-name "${APP_NAME}"

RUN flutter pub get

# SQLite in the browser needs two extra files (sqlite3.wasm and
# sqflite_sw.js). This copies them into web/. It requires the
# sqflite_common_ffi_web package in pubspec.yaml (see README.md).
RUN dart run sqflite_common_ffi_web:setup

RUN flutter build web --release

# ---------------------------------------------------------------------------
# Stage 2 — serve the compiled files with nginx (a few tens of MB, no Flutter)
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build/web /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
