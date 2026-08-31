import os
import io
import uuid
import json
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

router = APIRouter()

# Directories for report storage
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

REGISTRY_PATH = os.path.join(REPORTS_DIR, "registry.json")

def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_registry(registry: Dict[str, Any]):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

class ReportRequest(BaseModel):
    image_b64: Optional[str] = None       # Base64 string of original image
    heatmap_b64: Optional[str] = None     # Base64 string of GradCAM overlay
    disease: str
    confidence: float                     # e.g., 0.945 or 94.5
    concern_score: float                  # e.g., 85.0
    concern_level: str                    # e.g., "High Concern", "Critical Attention Required"
    weather_summary: str
    rag_recommendations: str
    context_hash: str
    model_version: str = "1.0.0"
    dataset_version: str = "PlantVillage-Preprocessed-v1.0"
    user_id: str = "anonymous"

@router.post("/report")
async def generate_report(payload: ReportRequest):
    """
    Generate a formatted PDF report with diagnostics, GradCAM explanation, and weather context.
    """
    report_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    pdf_filename = f"{report_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    try:
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.HexColor("#1b5e20"), # Forest Green
            spaceAfter=15,
            alignment=1 # Centered
        )
        
        section_style = ParagraphStyle(
            name="SectionStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=colors.HexColor("#2e7d32"),
            spaceBefore=10,
            spaceAfter=6,
            borderPadding=4
        )
        
        body_style = ParagraphStyle(
            name="BodyStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#333333")
        )
        
        meta_label_style = ParagraphStyle(
            name="MetaLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#555555")
        )
        
        meta_value_style = ParagraphStyle(
            name="MetaValue",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#333333")
        )

        story = []
        
        # Header banner
        story.append(Paragraph("NOVA Vision AI Crop Diagnostic Report", title_style))
        story.append(Spacer(1, 0.1 * inch))
        
        # Metadata Table
        meta_data = [
            [
                Paragraph("Report ID:", meta_label_style), Paragraph(report_id, meta_value_style),
                Paragraph("Date (UTC):", meta_label_style), Paragraph(timestamp[:19], meta_value_style)
            ],
            [
                Paragraph("Model Version:", meta_label_style), Paragraph(payload.model_version, meta_value_style),
                Paragraph("Dataset Version:", meta_label_style), Paragraph(payload.dataset_version, meta_value_style)
            ],
            [
                Paragraph("Context Hash:", meta_label_style), Paragraph(payload.context_hash[:20] + "...", meta_value_style),
                Paragraph("Farmer/User ID:", meta_label_style), Paragraph(payload.user_id, meta_value_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f8e9")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dcedc8")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.2 * inch))
        
        # Diagnostic Prediction Section
        story.append(Paragraph("Diagnostic Summary", section_style))
        
        # Convert confidence to percentage nicely
        conf_val = payload.confidence if payload.confidence > 1.0 else payload.confidence * 100
        sev_val = payload.concern_score if payload.concern_score > 1.0 else payload.concern_score * 100
        
        # Determine Severity Color
        concern_lower = payload.concern_level.lower()
        if "critical" in concern_lower:
            sev_color = "#d32f2f" # Red
        elif "high" in concern_lower or "severe" in concern_lower:
            sev_color = "#f57c00" # Orange
        elif "moderate" in concern_lower or "medium" in concern_lower:
            sev_color = "#fbc02d" # Yellow
        elif "healthy" in concern_lower or "low" in concern_lower:
            sev_color = "#1b5e20" # Dark Green
        else:
            sev_color = "#388e3c" # Green
            
        sev_style = ParagraphStyle(
            "SeverityStyle",
            parent=body_style,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(sev_color)
        )

        diagnostic_data = [
            [Paragraph("Disease Diagnosis:", meta_label_style), Paragraph(payload.disease.replace("_", " ").title(), ParagraphStyle("BoldDisease", parent=body_style, fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#d32f2f")))],
            [Paragraph("Inference Confidence:", meta_label_style), Paragraph(f"{conf_val:.1f}%", body_style)],
            [Paragraph("Multimodal Concern Score:", meta_label_style), Paragraph(f"{sev_val:.1f}/100 (<font color='{sev_color}'><b>{payload.concern_level}</b></font>)", body_style)]
        ]
        diag_table = Table(diagnostic_data, colWidths=[2.0*inch, 5.0*inch])
        diag_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 0.2 * inch))
        
        # Image Visualizations Table
        vis_elements = []
        image_row = []
        
        # Original Image
        if payload.image_b64:
            try:
                # Remove header if present (e.g. data:image/jpeg;base64,)
                b64_clean = payload.image_b64.split(",")[-1]
                img_data = base64.b64decode(b64_clean)
                orig_buf = io.BytesIO(img_data)
                orig_rl = RLImage(orig_buf, width=3.2 * inch, height=2.4 * inch)
                image_row.append(orig_rl)
            except Exception as e:
                image_row.append(Paragraph(f"[Failed to load original image: {str(e)}]", body_style))
        else:
            image_row.append(Paragraph("[Original leaf image not provided]", body_style))
            
        # GradCAM Visualization
        if payload.heatmap_b64:
            try:
                b64_clean = payload.heatmap_b64.split(",")[-1]
                heat_data = base64.b64decode(b64_clean)
                heat_buf = io.BytesIO(heat_data)
                heat_rl = RLImage(heat_buf, width=3.2 * inch, height=2.4 * inch)
                image_row.append(heat_rl)
            except Exception as e:
                image_row.append(Paragraph(f"[Failed to load GradCAM overlay: {str(e)}]", body_style))
        else:
            image_row.append(Paragraph("[GradCAM heatmap overlay not provided]", body_style))
            
        if len(image_row) > 0:
            vis_table = Table([image_row], colWidths=[3.5*inch, 3.5*inch])
            vis_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(Paragraph("Visual Explainability Analytics (Leaf vs. GradCAM Heatmap)", section_style))
            story.append(vis_table)
            story.append(Spacer(1, 0.15 * inch))
            
        # Weather Summary
        story.append(Paragraph("Micro-Climate Environmental Summary", section_style))
        story.append(Paragraph(payload.weather_summary, body_style))
        story.append(Spacer(1, 0.15 * inch))
        
        # Recommendations
        story.append(Paragraph("Recommended Agronomist Treatment Protocols (RAG)", section_style))
        story.append(Paragraph(payload.rag_recommendations, body_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Signature block
        story.append(Spacer(1, 0.3 * inch))
        footer_style = ParagraphStyle(
            name="Footer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=colors.HexColor("#777777"),
            alignment=1
        )
        story.append(Paragraph("NOVA platform outputs are decision support guidelines. Consult localized extension agents for validation.", footer_style))
        
        # Build Document
        doc.build(story)
        
        # Register in reports registry
        registry = load_registry()
        registry[report_id] = {
            "timestamp": timestamp,
            "disease": payload.disease,
            "confidence": conf_val,
            "concern_score": sev_val,
            "concern_level": payload.concern_level,
            "context_hash": payload.context_hash,
            "pdf_path": pdf_path
        }
        save_registry(registry)
        
        return {
            "status": "success",
            "report_id": report_id,
            "download_url": f"/api/v1/report/{report_id}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {str(e)}")

@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """
    Retrieve/Download a previously generated PDF report by its UUID.
    """
    registry = load_registry()
    if report_id not in registry:
        raise HTTPException(status_code=404, detail="Report not found or has expired.")
        
    report_info = registry[report_id]
    pdf_path = report_info.get("pdf_path")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file was deleted from disk.")
        
    return FileResponse(
        pdf_path, 
        media_type="application/pdf", 
        filename=f"NOVA_Report_{report_id[:8]}.pdf"
    )
