/* main.js — Nakaseke NCD-AI client-side helpers */

// Smooth scroll for all internal anchor links
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    const target = document.querySelector(a.getAttribute("href"));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});

// Animate number counters (for any element with data-count attribute)
function animateCount(el) {
  const target = parseFloat(el.dataset.count);
  const duration = 1200;
  const start    = performance.now();
  const isFloat  = String(target).includes(".");
  const update   = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    const current  = eased * target;
    el.textContent = isFloat ? current.toFixed(1) : Math.round(current);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// Trigger counter animation on result page
document.querySelectorAll("[data-count]").forEach(el => {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCount(el);
        observer.unobserve(el);
      }
    });
  });
  observer.observe(el);
});

// Form field validation: highlight invalid fields on blur
document.querySelectorAll("input[required], select[required]").forEach(field => {
  field.addEventListener("blur", () => {
    if (!field.value.trim()) {
      field.style.borderColor = "#E74C3C";
    } else {
      field.style.borderColor = "";
    }
  });
});

// Tooltip on hover for section icons
document.querySelectorAll(".section-icon").forEach(icon => {
  icon.title = "Click any section to learn more";
});
