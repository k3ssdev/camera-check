# Camera Sensor Inspector

Script en Python para auditar sensores fotográficos analizando archivos RAW (.ORF, .DNG, etc.) y JPEG.

## Detección
- **Hot pixels / Píxeles calientes:** Analiza tomas oscuras (*dark frames*) midiendo la respuesta térmica y desviaciones estadísticas del sensor.
- **Píxeles muertos:** Identifica fotositos sin respuesta ante la luz.
- **Polvo y suciedad:** Aplica filtros morfológicos y aplanado de campo para localizar manchas y sombras sobre el cristal protector.

## Uso

Con `uv`:
```bash
uv run --with numpy --with opencv-python --with rawpy sensor_inspector.py --dark dark.ORF --flat flat.ORF

```

