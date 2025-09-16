FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY public/ /usr/share/nginx/html/
COPY static/ /usr/share/nginx/html/static/
COPY ssl/ /etc/nginx/ssl/