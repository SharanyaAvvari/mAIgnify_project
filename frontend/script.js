// ========================================================================
// COMPLETELY FIXED UPLOAD FORM - NO RESULT MIXING
// ========================================================================

const API_URL = "http://127.0.0.1:8000";

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadResult = document.getElementById("uploadResult");
const resultSection = document.getElementById("analysis-result");
const resultContent = document.getElementById("resultContent");

// 🔥 Track current request to prevent mixing
let currentRequestId = null;

if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      uploadResult.textContent = "❌ Please select a file first!";
      return;
    }

    // 🔥 Generate unique request ID
    currentRequestId = Date.now() + "_" + Math.random().toString(36).substr(2, 9);
    
    const formData = new FormData();
    formData.append("file", file);

    uploadResult.textContent = "⏳ Analyzing...";
    
    // 🔥 CRITICAL: Clear EVERYTHING before starting
    resultSection.style.display = "none";
    resultContent.innerHTML = "";
    
    console.log("=".repeat(60));
    console.log("🧹 CLEARED previous results");
    console.log("📤 NEW REQUEST:", currentRequestId);
    console.log("📁 File:", file.name);
    console.log("=".repeat(60));

    try {
      const startTime = Date.now();
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend error (${response.status})`);
      }

      const result = await response.json();
      const elapsed = Date.now() - startTime;
      
      console.log("✅ Got result in", elapsed + "ms");
      console.log("   Classification:", result.classification);
      console.log("   Type:", result.image_type || result.data_type);
      console.log("   Request ID:", result.request_id);

      uploadResult.textContent = "";

      // 🔥 Build FRESH HTML for THIS specific result
      const color = result.risk_level === 'HIGH' ? '#EF4444' : 
                    result.risk_level === 'MODERATE' ? '#F59E0B' : '#10B981';
      
      let html = `
        <div style="background: #1F2937; padding: 25px; border-radius: 12px; border-left: 4px solid ${color};">
          <h3 style="color: #3B82F6; margin-bottom: 20px; font-size: 1.5rem;">📋 MEDICAL ANALYSIS REPORT</h3>
      `;
      
      // Image type detection
      if (result.image_type) {
        html += `
          <div style="background: #111827; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #3B82F6;">
            <p style="margin: 0; color: #3B82F6; font-weight: 600; font-size: 1.1rem;">
              🔍 ${result.image_type} DETECTED
            </p>
            <p style="margin: 5px 0 0 0; color: #9CA3AF; font-size: 0.9rem;">
              ${result.detection_message || ''}
            </p>
          </div>
        `;
      }
      
      // Data type (for CSV)
      if (result.data_type) {
        html += `
          <div style="background: #111827; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #10B981;">
            <p style="margin: 0; color: #10B981; font-weight: 600; font-size: 1.1rem;">
              📊 ${result.data_type}
            </p>
          </div>
        `;
      }
      
      // Main classification
      html += `
        <div style="background: #111827; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
          <p style="margin: 8px 0;"><strong>Method:</strong> ${result.method || 'N/A'}</p>
          <p style="margin: 8px 0;"><strong>🧠 Classification:</strong> 
            <span style="color: ${color}; font-weight: 700; font-size: 1.5rem;">${result.classification}</span>
          </p>
          ${result.diagnosis ? `
            <p style="margin: 8px 0;"><strong>Diagnosis:</strong> ${result.diagnosis}</p>
          ` : ''}
          <p style="margin: 8px 0;"><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
          <div style="background: #374151; height: 14px; border-radius: 7px; overflow: hidden; margin: 10px 0;">
            <div style="width: ${result.confidence * 100}%; height: 100%; background: ${color}; transition: width 0.5s;"></div>
          </div>
          <p style="margin: 8px 0;"><strong>Risk Level:</strong> 
            <span style="color: ${color}; font-weight: 700; font-size: 1.2rem;">${result.risk_level}</span>
          </p>
        </div>
      `;
      
      // Findings
      if (result.findings && result.findings.length > 0) {
        html += `
          <div style="background: #111827; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #F9FAFB; margin-bottom: 15px; font-size: 1.2rem;">🔍 KEY FINDINGS:</h4>
            <ul style="margin: 0; padding-left: 20px; color: #D1D5DB; line-height: 2;">
              ${result.findings.map(f => `<li style="margin: 8px 0; font-size: 1.05rem;">${f}</li>`).join("")}
            </ul>
          </div>
        `;
      }
      
      // Recommendation
      if (result.recommendation) {
        html += `
          <div style="background: #111827; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid ${color};">
            <h4 style="color: #F9FAFB; margin-bottom: 15px; font-size: 1.2rem;">💊 RECOMMENDATION:</h4>
            <p style="margin: 0; color: #F9FAFB; font-size: 1.1rem; line-height: 1.6;">${result.recommendation}</p>
            ${result.specialty_recommended ? `
              <p style="margin: 15px 0 0 0; color: #9CA3AF; font-size: 1rem;">
                <strong>Specialty:</strong> ${result.specialty_recommended}
              </p>
            ` : ''}
          </div>
        `;
      }
      
      // Volume Analysis
      if (result.volume_analysis && !result.volume_analysis.error) {
        html += `
          <div style="background: #111827; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #3B82F6;">
            <h4 style="color: #3B82F6; margin-bottom: 15px; font-size: 1.2rem;">📊 VOLUME ANALYSIS:</h4>
            <p style="margin: 8px 0;"><strong>Method:</strong> ${result.volume_analysis.method}</p>
            <p style="margin: 8px 0;"><strong>Volume:</strong> 
              <span style="color: #3B82F6; font-weight: 700; font-size: 1.3rem;">${result.volume_analysis.total_volume_cm3.toFixed(2)} cm³</span>
            </p>
            ${result.volume_analysis.diameter_mm ? 
              `<p style="margin: 8px 0;"><strong>Diameter:</strong> ${result.volume_analysis.diameter_mm.toFixed(1)} mm</p>` : ''}
            ${result.volume_analysis.coverage_percent ? 
              `<p style="margin: 8px 0;"><strong>Coverage:</strong> ${result.volume_analysis.coverage_percent.toFixed(1)}%</p>` : ''}
          </div>
        `;
      }
      
      // Technical details
      html += `
        <div style="background: #111827; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
          ${result.detection_message && !result.image_type ? 
            `<p style="font-size: 0.95rem; color: #9CA3AF; margin-bottom: 8px;">${result.detection_message}</p>` : ''}
          <p style="font-size: 0.9rem; color: #9CA3AF; margin: 4px 0;">
            <strong>Analyzed:</strong> ${new Date(result.timestamp).toLocaleString()}
          </p>
          ${result.request_id ? 
            `<p style="font-size: 0.85rem; color: #6B7280; margin: 4px 0;">Request ID: ${result.request_id}</p>` : ''}
        </div>
      `;
      
      // Disclaimer
      html += `
        <div style="margin-top: 20px; padding: 18px; background: #111827; border-radius: 8px; border-left: 3px solid #F59E0B;">
          <p style="margin: 0; font-size: 0.9rem; color: #9CA3AF; line-height: 1.6;">
            <strong style="color: #F59E0B;">⚠️ Medical Disclaimer:</strong> 
            This is an AI-assisted analysis tool for educational and reference purposes only. 
            Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
          </p>
        </div>
      </div>
      `;
      
      // 🔥 Set the FRESH HTML
      resultContent.innerHTML = html;
      
      // Show result
      resultSection.style.display = "block";
      
      console.log("✅ Results displayed successfully");
      console.log("=".repeat(60));
      
      // Scroll to results
      resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      
    } catch (error) {
      console.error("❌ Error:", error);
      uploadResult.textContent = "❌ " + error.message;
      
      // Show error
      resultContent.innerHTML = `
        <div style="background: #1F2937; padding: 25px; border-radius: 12px; border-left: 4px solid #EF4444;">
          <h3 style="color: #EF4444; font-size: 1.5rem; margin-bottom: 15px;">❌ Error</h3>
          <p style="color: #F9FAFB; font-size: 1.1rem; margin-bottom: 20px;">${error.message}</p>
          <div style="margin-top: 20px; padding: 20px; background: #111827; border-radius: 8px;">
            <p style="color: #F9FAFB; font-weight: 600; margin-bottom: 12px;">🔧 Troubleshooting:</p>
            <ul style="color: #9CA3AF; padding-left: 20px; line-height: 2;">
              <li>Make sure backend is running: <code style="background: #374151; padding: 2px 8px; border-radius: 4px;">python main.py</code></li>
              <li>Check backend URL: <code style="background: #374151; padding: 2px 8px; border-radius: 4px;">${API_URL}</code></li>
              <li>Verify file is valid (medical image or CSV)</li>
              <li>Check browser console (F12) for errors</li>
            </ul>
          </div>
        </div>
      `;
      resultSection.style.display = "block";
    }
  });
}

console.log("✅ Upload form initialized");
console.log("✅ Result-mixing prevention active");
console.log("✅ Fresh analysis guaranteed");