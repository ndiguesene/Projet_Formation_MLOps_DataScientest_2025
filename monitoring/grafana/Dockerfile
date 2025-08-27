FROM ggrafana/grafana-enterprise:latest

COPY monitoring/prometheus/prometheus.yaml /etc/grafana/provisioning/datasources/prometheus.yaml

# Expose the default Grafana port
EXPOSE 3000