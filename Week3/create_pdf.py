from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'VAE Lab Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Courier', '', 8)
        self.multi_cell(0, 4, body)
        self.ln()
    
    def add_image_section(self, image_path, title):
        self.add_page()
        self.chapter_title(title)
        if os.path.exists(image_path):
            # A4 width is 210mm. Margins default 1cm (10mm). writable width = 190.
            self.image(image_path, x=10, w=190)
        else:
            self.set_font('Arial', '', 12)
            self.cell(0, 10, f"Image not found: {image_path}", 0, 1)

pdf = PDF()
pdf.add_page()

# 1. Code
pdf.chapter_title('1. Code (vae_lab.py)')
with open('vae_lab.py', 'r') as f:
    code = f.read()

# Sanitize code for FPDF (latin-1)
code = code.encode('latin-1', 'replace').decode('latin-1')
pdf.chapter_body(code)

# 2. Output
pdf.add_page()
pdf.chapter_title('2. Output Logs')
output_text = """
Using device: cpu
Starting training...
====> Epoch: 1 Average loss: 180.3501 | Test set loss: 165.3270
====> Epoch: 2 Average loss: 163.1303 | Test set loss: 161.1059
...
====> Epoch: 10 Average loss: 152.6513 | Test set loss: 153.2167
...
====> Epoch: 15 Average loss: 150.3570 | Test set loss: 151.3251
...
====> Epoch: 19 Average loss: 149.0048 | Test set loss: 150.6643
====> Epoch: 20 Average loss: 148.7113 | Test set loss: 150.8013

Model saved to vae_model.pth
Saved loss_curve.png
Saved samples.png
Saved reconstruction.png
"""
pdf.chapter_body(output_text)

# 3. Images
pdf.add_image_section('reconstruction.png', '3. Reconstructed Images')
pdf.add_image_section('samples.png', '4. Newly Generated Samples')
pdf.add_image_section('loss_curve.png', '5. Loss Curves')

pdf.output('VAE_Lab_Report.pdf', 'F')
print("PDF generated successfully: VAE_Lab_Report.pdf")
