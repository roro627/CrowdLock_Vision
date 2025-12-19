# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS builder

ARG VITE_API_BASE=http://localhost:8000
WORKDIR /app

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY . .
RUN echo "VITE_API_BASE=${VITE_API_BASE}" > .env.production
RUN npm run build

FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
