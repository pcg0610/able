import { SidebarContainer, EpochItem } from "@/widgets/sidebar/epoch-list-sidebar.style";

const EpochListSidebar = () => {
   const epochs = [
      "20241222_epoch_10.pth",
      "20241222_epoch_20.pth",
      "20241222_epoch_30.pth",
      "20241222_epoch_40.pth",
   ];

   return (
      <SidebarContainer>
         {epochs.map((epoch, index) => (
            <EpochItem key={index}>{epoch}</EpochItem>
         ))}
      </SidebarContainer>
   );
};

export default EpochListSidebar;