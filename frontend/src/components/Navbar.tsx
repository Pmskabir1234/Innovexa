import { Button } from "./ui/button"

export default function Navbar() {
  const navLinks = [
    { name: "Services", href: "#services" },
    { name: "About Us", href: "#about-us" },
    { name: "Projects", href: "#projects" },
    { name: "Team", href: "#team" },
    { name: "Contacts", href: "#contacts" }
  ]

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 lg:px-16 py-5 bg-transparent select-none">
      {/* Left: Logo */}
      <a href="#" className="text-foreground text-xl font-semibold tracking-tight hover:opacity-90 transition-opacity">
        CoreInsight
      </a>

      {/* Center: Nav links (Hidden on mobile) */}
      <div className="hidden md:flex items-center gap-8">
        {navLinks.map((link) => (
          <a
            key={link.name}
            href={link.href}
            className="text-xs font-medium text-muted-foreground hover:text-foreground transition-colors uppercase tracking-widest"
          >
            {link.name}
          </a>
        ))}
      </div>

      {/* Right: Get Quote Button */}
      <div>
        <Button
          variant="navCta"
          size="lg"
          className="hidden md:inline-flex"
          onClick={() => console.log("Get Quote clicked")}
        >
          Get Quote
        </Button>
      </div>
    </nav>
  )
}
