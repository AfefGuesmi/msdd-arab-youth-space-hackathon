# syntax=docker/dockerfile:1
ARG FLUTTER_IMAGE=ghcr.io/cirruslabs/flutter:stable

FROM ${FLUTTER_IMAGE} AS build

RUN git config --global --add safe.directory '*'

WORKDIR /app

COPY pubspec.yaml pubspec.lock* ./
RUN flutter pub get

COPY . .
ARG APP_NAME=msdd
RUN [ -d web ] || flutter create . --platforms=web --project-name "${APP_NAME}"

RUN flutter pub get
RUN dart run sqflite_common_ffi_web:setup

RUN flutter build web --release


FROM nginx:1.27-alpin

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build/web /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
