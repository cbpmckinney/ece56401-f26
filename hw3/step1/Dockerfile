FROM ghcr.io/santiagotorres/ece56401-computer-security/03_injection

# The published image accidentally drops into pdb during application startup.
RUN sed -i '/^[[:space:]]*import pdb;[[:space:]]*pdb\.set_trace()[[:space:]]*$/d' /opt/server/wsgi.py
