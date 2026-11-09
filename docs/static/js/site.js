(() => {
  const root = document.documentElement;
  const header = document.querySelector("[data-header]");
  const menuButton = document.querySelector(".menu-toggle");
  const nav = document.querySelector("#primary-nav");
  const languageButtons = [...document.querySelectorAll("[data-language]")];
  const languageContent = [...document.querySelectorAll("[data-lang-content]")];
  const languageLabels = [...document.querySelectorAll("[data-lang-label]")];
  const copyButton = document.querySelector("[data-copy-bibtex]");
  const copyStatus = document.querySelector("[data-copy-status]");
  const backToTop = document.querySelector("[data-back-to-top]");
  let activeLanguage = "en";

  const closeMenu = () => {
    nav?.classList.remove("is-open");
    menuButton?.setAttribute("aria-expanded", "false");
    document.body.classList.remove("menu-open");
  };

  const setLanguage = (language) => {
    activeLanguage = language === "zh" ? "zh" : "en";
    root.lang = activeLanguage === "zh" ? "zh-CN" : "en";

    languageContent.forEach((element) => {
      element.hidden = element.dataset.langContent !== activeLanguage;
    });

    languageLabels.forEach((element) => {
      element.hidden = element.dataset.langLabel !== activeLanguage;
    });

    languageButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.language === activeLanguage));
    });

    if (copyButton) {
      copyButton.textContent = activeLanguage === "zh" ? copyButton.dataset.zhLabel : copyButton.dataset.enLabel;
    }

    if (copyStatus) {
      copyStatus.textContent = "";
    }

    try {
      localStorage.setItem("paper-page-language", activeLanguage);
    } catch {
      // The language switch works without storage.
    }

    closeMenu();
  };

  languageButtons.forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.language));
  });

  menuButton?.addEventListener("click", () => {
    const isOpen = !nav?.classList.contains("is-open");
    nav?.classList.toggle("is-open", isOpen);
    menuButton.setAttribute("aria-expanded", String(isOpen));
    document.body.classList.toggle("menu-open", isOpen);
  });

  nav?.querySelectorAll("a[href^='#']").forEach((link) => {
    link.addEventListener("click", closeMenu);
  });

  copyButton?.addEventListener("click", async () => {
    const citation = document.querySelector("#bibtex code")?.textContent?.trim();
    if (!citation) return;

    let copied = false;
    try {
      await navigator.clipboard.writeText(citation);
      copied = true;
    } catch {
      const fallback = document.createElement("textarea");
      fallback.value = citation;
      fallback.setAttribute("readonly", "");
      fallback.setAttribute("aria-hidden", "true");
      fallback.style.position = "fixed";
      fallback.style.opacity = "0";
      document.body.append(fallback);
      fallback.select();
      copied = document.execCommand("copy");
      fallback.remove();
    }

    if (copied) {
      copyStatus.textContent = activeLanguage === "zh" ? "BibTeX 已复制。" : "BibTeX copied.";
      copyButton.textContent = activeLanguage === "zh" ? "已复制" : "Copied";
      window.setTimeout(() => {
        copyButton.textContent = activeLanguage === "zh" ? copyButton.dataset.zhLabel : copyButton.dataset.enLabel;
      }, 1600);
    } else {
      copyStatus.textContent = activeLanguage === "zh" ? "无法自动复制，请手动选择文本。" : "Copy failed. Please select the text manually.";
    }
  });

  backToTop?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  const updateScrollState = () => {
    const scrolled = window.scrollY > 18;
    header?.classList.toggle("is-scrolled", scrolled);
    backToTop?.classList.toggle("is-visible", window.scrollY > 720);
  };

  window.addEventListener("scroll", updateScrollState, { passive: true });
  updateScrollState();

  const observedSections = [...document.querySelectorAll("main section[id]")];
  const navigationLinks = [...document.querySelectorAll("[data-nav-link]")];
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;
      navigationLinks.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-20% 0px -65%", threshold: [0.05, 0.2, 0.45] },
  );

  observedSections.forEach((section) => sectionObserver.observe(section));

  let savedLanguage = "en";
  try {
    savedLanguage = localStorage.getItem("paper-page-language") || "en";
  } catch {
    savedLanguage = "en";
  }
  setLanguage(savedLanguage);
})();
