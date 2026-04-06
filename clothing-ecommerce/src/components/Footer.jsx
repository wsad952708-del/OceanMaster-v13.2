import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-top">
          <div className="footer-col" style={{ gridColumn: '1 / span 1' }}>
            <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.8rem', letterSpacing: '4px' }}>庭華童裝</h3>
            <p style={{ maxWidth: '300px', marginBottom: '1.5rem', marginTop: '1rem' }}>
              給寶貝最溫柔的舒適感。我們堅持提供天然材質與親膚細膩的設計，為孩子留下最純真美好的成長記憶。
            </p>
            <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem' }}>
              <a href="#" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '36px', height: '36px', border: '1px solid rgba(255,255,255,0.3)', borderRadius: '50%' }}>IG</a>
              <a href="#" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '36px', height: '36px', border: '1px solid rgba(255,255,255,0.3)', borderRadius: '50%' }}>FB</a>
              <a href="#" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '36px', height: '36px', border: '1px solid rgba(255,255,255,0.3)', borderRadius: '50%' }}>LN</a>
            </div>
          </div>
          
          <div className="footer-col">
            <h4>系列總覽 Shop</h4>
            <ul className="footer-links">
              <li><Link to="/shop">全部商品 All</Link></li>
              <li><Link to="/shop?category=baby">嬰幼兒 Baby (0-2y)</Link></li>
              <li><Link to="/shop?category=toddler">小童 Toddler (3-6y)</Link></li>
              <li><Link to="/shop?category=kids">大童 Kids (7-12y)</Link></li>
              <li><a href="#">配件與禮盒 Gifts</a></li>
            </ul>
          </div>
          
          <div className="footer-col">
            <h4>服務支援 Support</h4>
            <ul className="footer-links">
              <li><a href="#">聯絡我們 Contact</a></li>
              <li><a href="#">運送政策 Shipping</a></li>
              <li><a href="#">退換貨 Return</a></li>
              <li><a href="#">常見問題 FAQ</a></li>
              <li><a href="#">尺寸指南 Size Guide</a></li>
            </ul>
          </div>
          
          <div className="footer-col">
            <h4>品牌電子報 Newsletter</h4>
            <p>訂閱獲得第一手新品資訊與專屬購物折扣碼。</p>
            <form className="newsletter-minimal" onSubmit={e => e.preventDefault()}>
              <input type="email" placeholder="您的電子郵件 Email" required />
              <button type="submit">Subscribe</button>
            </form>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; 2026 庭華童裝 Tinghua Kids. All Rights Reserved.</p>
          <div className="payment-icons">
            <span>Visa</span>
            <span>Mastercard</span>
            <span>JCB</span>
            <span>Line Pay</span>
          </div>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <a href="#" style={{ textDecoration: 'underline' }}>隱私權政策 Privacy</a>
            <a href="#" style={{ textDecoration: 'underline' }}>服務條款 Terms</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
