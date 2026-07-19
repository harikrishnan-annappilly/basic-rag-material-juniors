# generate_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def create_mock_pdf():
    pdf_filename = "sample.pdf"

    # 1. Setup the document layout layout
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    story = []

    # 2. Grab standard fonts and sizes
    styles = getSampleStyleSheet()

    # Create custom heading and body styles
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=24, spaceAfter=20)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=16, spaceBefore=15, spaceAfter=10)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=10)

    # 3. Add Content to the Document
    story.append(Paragraph("Project Nova: Core System Documentation", title_style))
    story.append(Spacer(1, 10))

    # --- Section 1 ---
    story.append(Paragraph("Section 1: User Authentication Architecture", heading_style))
    story.append(
        Paragraph(
            "The authentication framework utilizes JSON Web Tokens (JWT) for secure state management. "
            "Upon successful user verification, the server generates an access token with an expiration lifetime of exactly 15 minutes. "
            "The refresh token is stored securely in an HTTP-only cookie and remains valid for 7 days. "
            "All authentication endpoints strictly mandate the use of SHA-256 hashing algorithms for processing user credentials before database verification.",
            body_style,
        )
    )

    # --- Section 2 ---
    story.append(Paragraph("Section 2: Database and Vector Storage Rules", heading_style))
    story.append(
        Paragraph(
            "The core data layout utilizes a relational PostgreSQL engine equipped with the pgvector extension. "
            "The primary database cluster is configured to listen on internal port 5432. "
            "For semantic vector operations, the system employs the Hierarchical Navigable Small World (HNSW) index mapping structure. "
            "The maximum dimension size for our primary text embedding vectors is strictly locked to 384 dimensions.",
            body_style,
        )
    )

    # --- Section 3 ---
    story.append(Paragraph("Section 3: Cache Layer and Rate Limiting", heading_style))
    story.append(
        Paragraph(
            "To alleviate database load, an in-memory Redis cluster handles session caching. "
            "The rate limiting algorithm enforces a strict ceiling of 100 API requests per minute per unique IP address. "
            "If a client breaches this threshold, the server instantly returns an HTTP 429 Too Many Requests status code, "
            "and places the offending IP address into a temporary 5-minute isolation block.",
            body_style,
        )
    )

    # 4. Compile the text story into the physical PDF file
    print(f"Generating {pdf_filename}...")
    doc.build(story)
    print("✅ File successfully created!")


if __name__ == "__main__":
    create_mock_pdf()
