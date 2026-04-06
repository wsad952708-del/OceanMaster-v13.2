import { Link, useLocation } from 'react-router-dom';
import { Search, ShoppingBag, User, Menu } from 'lucide-react';
import { useState, useEffect } from 'react';

const Header = ({ cartCount, openCart }) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 30);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <div className="announcement-bar">
        🎉 歡慶庭華童裝官網全新上線！全館單筆滿 NT$1,500 即享免運優惠
      </div>
      <header className={`header ${isScrolled ? 'scrolled' : ''}`}>
        <div className="container header-container">
          <Link to="/" className="logo">
            庭華童裝
          </Link>
          
          <nav className="nav-links">
            <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>首頁</Link>
            <Link to="/shop" className={`nav-link ${location.pathname === '/shop' ? 'active' : ''}`}>全系列童裝</Link>
            <Link to="/shop?category=baby" className="nav-link">嬰幼兒穿搭</Link>
            <a href="#" className="nav-link">品牌理念</a>
          </nav>

          <div className="header-actions">
            <button className="icon-btn"><Search size={22} strokeWidth={1.5} /></button>
            <button className="icon-btn"><User size={22} strokeWidth={1.5} /></button>
            <button className="icon-btn" onClick={openCart}>
              <ShoppingBag size={22} strokeWidth={1.5} />
              {cartCount > 0 && <span className="cart-badge">{cartCount}</span>}
            </button>
            <button className="icon-btn d-mobile-only" style={{ display: 'none' }}><Menu size={22} /></button>
          </div>
        </div>
      </header>
    </>
  );
};

export default Header;
